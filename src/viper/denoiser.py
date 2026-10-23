import os
import copy
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class AFMPairDataset(torch.utils.data.Dataset):
    def __init__(self, x_cuts, y_cuts, mx_cuts=None, my_cuts=None, augment=True):
        assert x_cuts.shape[0] == y_cuts.shape[0]
        if mx_cuts is not None:
            assert mx_cuts.shape[0] == x_cuts.shape[0]
        if my_cuts is not None:
            assert my_cuts.shape[0] == y_cuts.shape[0]

        self.x_cuts = x_cuts
        self.y_cuts = y_cuts
        self.mx_cuts = mx_cuts
        self.my_cuts = my_cuts
        self.augment = augment
        self.has_masks = (mx_cuts is not None) and (my_cuts is not None)
        # 0: identity, 1: horizontal flip, 2: vertical flip, 3: 180 rotation
        self._ops = [0, 1, 2, 3]

    def __len__(self):
        return self.x_cuts.shape[0]

    @staticmethod
    def _apply_transform(img, op):
        # img shape is (C, H, W)
        if op == 1: # horizontal flip (width)
            return torch.flip(img, dims=(-1,))           
        if op == 2: # vertical flip (height)
            return torch.flip(img, dims=(-2,))           
        if op == 3: # 180 degree rotation
            return torch.rot90(img, k=2, dims=(-2, -1))  
        return img  # identity

    def __getitem__(self, idx):
        x = self.x_cuts[idx]
        y = self.y_cuts[idx]

        if self.has_masks:
            mx = self.mx_cuts[idx]
            my = self.my_cuts[idx]

        if self.augment:
            op = random.choice(self._ops)
            x = self._apply_transform(x, op)
            y = self._apply_transform(y, op)
            if self.has_masks:
                mx = self._apply_transform(mx, op)
                my = self._apply_transform(my, op)

        if self.has_masks:
            return x, y, mx, my
        return x, y

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, activation="relu", double=False):
        super().__init__()
        act = {"relu": nn.ReLU, "leaky_relu": lambda: nn.LeakyReLU(0.01)}[activation]
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size, padding="same"), act()]
        if double:
            layers += [nn.Conv2d(out_ch, out_ch, kernel_size, padding="same"), act()]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, activation="relu", kernel_size=3, stride=2):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=kernel_size, stride=stride,
            padding=kernel_size // 2, output_padding=stride - 1)
        self.conv = ConvBlock(
            out_ch + skip_ch, out_ch, activation=activation, double=True)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class SelfAttention2d(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        residual = x
        x = self.norm(x)
        tokens = x.flatten(2).transpose(1, 2)
        attended, _ = self.attn(tokens, tokens, tokens)
        attended = attended.transpose(1, 2).reshape(b, c, h, w)
        return residual + attended

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden     = int(dim*mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        a, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x

class N2N(nn.Module):
    def __init__(self, channels=1, base_filters=48, depth=5, activation="leaky_relu", img_size=32):
        super().__init__()
        max_depth = max(1, int(math.log2(img_size)) - 2)
        if depth > max_depth:
            print(f"N2N: reducing depth from {depth} to {max_depth} to avoid collapse (img_size={img_size})")
            depth = max_depth
        self.depth = depth
        self.channels = channels

        self.pool = nn.MaxPool2d(2, ceil_mode=True)
        self.stem = nn.Sequential(
            ConvBlock(channels, base_filters, activation=activation),
            ConvBlock(base_filters, base_filters, activation=activation),
        )
        self.enc_blocks = nn.ModuleList([
            ConvBlock(base_filters, base_filters, activation=activation)
            for _ in range(depth)
        ])
        self.bottleneck = ConvBlock(base_filters, base_filters, activation=activation)

        # Skip channel list, deepest-first
        skip_channels = [base_filters] * depth + [channels]

        dec_filters = base_filters*2
        self.decoders = nn.ModuleList()
        in_ch = base_filters
        for skip_ch in skip_channels:
            self.decoders.append(
                DecoderBlock(
                    in_ch, skip_ch, dec_filters,
                    activation=activation,
                    kernel_size=2, stride=2)
            )
            in_ch = dec_filters

        self.out_conv = nn.Conv2d(dec_filters, channels, 3, padding="same")

    def forward(self, x):
        h = self.stem(x)

        enc_skips = []
        for enc in self.enc_blocks:
            h = self.pool(h)
            h = enc(h)
            enc_skips.append(h)

        h = self.pool(h)
        h = self.bottleneck(h)

        # deepest-first encoder skips, then the raw input as the last skip
        skip_order = list(reversed(enc_skips)) + [x]

        for dec, skip in zip(self.decoders, skip_order):
            h = dec(h, skip)

        out = self.out_conv(h)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="nearest")
        return out

class UNet(nn.Module):
    def __init__(
            self, channels=1, filters=(16, 32, 64, 128, 256),
            bottleneck_filters=512, activation="relu"):
        super().__init__()
        self.pool = nn.MaxPool2d(2, ceil_mode=True)

        enc_in = channels
        self.encoders = nn.ModuleList()
        for f in filters:
            self.encoders.append(ConvBlock(enc_in, f, activation=activation, double=True))
            enc_in = f

        self.bottleneck = nn.Sequential(
            ConvBlock(filters[-1], bottleneck_filters, activation=activation),
            ConvBlock(bottleneck_filters, bottleneck_filters, activation=activation),
        )

        rev_filters = list(reversed(filters))
        self.decoders = nn.ModuleList()
        in_ch = bottleneck_filters
        for f in rev_filters:
            self.decoders.append(DecoderBlock(
                in_ch, f, f,
                activation=activation,
                kernel_size=3, stride=2))
            in_ch = f

        self.out_conv = nn.Conv2d(rev_filters[-1], channels, 3, padding="same")

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)

        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        return self.out_conv(x)

class ViTDenoiser(nn.Module):
    def __init__(
            self, img_size=32, channels=1, patch_size=4, dim=192,
            depth=6, num_heads=6, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.channels = channels
        self.grid = img_size // patch_size
        num_patches = self.grid**2
        patch_dim = channels*patch_size * patch_size

        self.patch_embed = nn.Linear(patch_dim, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

        # simple linear decoder: project each token back to a pixel patch
        self.head = nn.Linear(dim, patch_dim)

    def _patchify(self, x):
        b, c, h, w = x.shape
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p) # (B, C, gh, gw, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(b, self.grid * self.grid, c * p * p)
        return x

    def _unpatchify(self, tokens, batch_size):
        p = self.patch_size
        c = self.channels
        g = self.grid
        x = tokens.view(batch_size, g, g, c, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(batch_size, c, g * p, g * p)
        return x

    def forward(self, x):
        b, c, h, w = x.shape
        if h != self.img_size or w != self.img_size:
            raise ValueError(
                f"ViTDenoiser was built with img_size={self.img_size} but received {h}x{w}.")
        tokens = self.patch_embed(self._patchify(x)) + self.pos_embed
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        patches_out = self.head(tokens)
        return self._unpatchify(patches_out, b)

class AttnUNet(nn.Module):
    def __init__(
            self, channels=1, filters=(16, 32, 64, 128, 256),
            bottleneck_filters=512, activation="relu",
            attn_heads=4, attn_from_stage=2):
        super().__init__()
        self.pool = nn.MaxPool2d(2, ceil_mode=True)
        self.attn_from_stage = attn_from_stage

        enc_in = channels
        self.encoders = nn.ModuleList()
        self.enc_attn = nn.ModuleList()
        for i, f in enumerate(filters):
            self.encoders.append(
                ConvBlock(enc_in, f, activation=activation, double=True))
            self.enc_attn.append(
                SelfAttention2d(f, attn_heads) if i >= attn_from_stage else nn.Identity())
            enc_in = f

        self.bottleneck_conv = ConvBlock(filters[-1], bottleneck_filters, activation=activation)
        self.bottleneck_attn = SelfAttention2d(bottleneck_filters, attn_heads)

        rev_filters = list(reversed(filters))
        self.decoders = nn.ModuleList()
        self.dec_attn = nn.ModuleList()
        in_ch = bottleneck_filters
        n = len(rev_filters)
        for i, f in enumerate(rev_filters):
            self.decoders.append(DecoderBlock(
                in_ch, f, f,
                activation=activation,
                kernel_size=3, stride=2))
            # deepest decoder stages (closest to bottleneck)
            stage_depth_from_bottleneck = n - 1 - i
            self.dec_attn.append(
                SelfAttention2d(f, attn_heads) if stage_depth_from_bottleneck >= attn_from_stage
                else nn.Identity()
            )
            in_ch = f

        self.out_conv = nn.Conv2d(rev_filters[-1], channels, 3, padding="same")

    def forward(self, x):
        skips = []
        for enc, attn in zip(self.encoders, self.enc_attn):
            x = enc(x)
            x = attn(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck_conv(x)
        x = self.bottleneck_attn(x)

        for dec, attn, skip in zip(self.decoders, self.dec_attn, reversed(skips)):
            x = dec(x, skip)
            x = attn(x)

        return self.out_conv(x)

class Denoiser:
    def __init__(self, name="Denoiser", device="cpu"):
        self.name = name
        self.device = torch.device(device)
        self.model = None

    def set_dataset(self, data):
        self.dataset = data

    def split(self, fractions=(0.7, 0.2, 0.1), norm=False):
        n = self.dataset.cuts_num
        train_size, train_val_size, _ = (np.array(fractions)*n).astype(int)
        train_i, train_val_i = train_size, train_size + train_val_size

        ids = list(range(n))
        random.shuffle(ids)
        self.train_ids = ids[0:train_i]
        self.train_val_ids = ids[train_i:train_val_i]
        self.test_ids = ids[train_val_i:n]

        self.norm = norm
        x = self.dataset.x_cuts if not norm else self.dataset.x_norm
        y = self.dataset.y_cuts if not norm else self.dataset.y_norm
        mx = self.dataset.mx_cuts
        my = self.dataset.my_cuts

        self.train_X  = x[self.train_ids]
        self.train_y  = y[self.train_ids]
        self.train_MX = mx[self.train_ids]
        self.train_My = my[self.train_ids]

        # same-direction held-out set (X -> Y), for monitoring
        self.train_val_X  = x[self.train_val_ids]
        self.train_val_y  = y[self.train_val_ids]
        self.train_val_MX = mx[self.train_val_ids]
        self.train_val_My = my[self.train_val_ids]

        # real validation: reversed direction (Y -> X) on the SAME crops
        self.val_X  = y[self.train_val_ids]
        self.val_y  = x[self.train_val_ids]
        self.val_MX = my[self.train_val_ids]
        self.val_My = mx[self.train_val_ids]

        self.test_X  = x[self.test_ids]
        self.test_y  = y[self.test_ids]
        self.test_MX = mx[self.test_ids]
        self.test_My = my[self.test_ids]

    def build_model(
            self, arch="unet", width=None, height=None, channels=None,
            lr=1e-3, loss="mse", summary=False, **model_kwargs):
        if arch not in MODEL_REGISTRY:
            raise ValueError(f"Unknown arch '{arch}'. Options: {list(MODEL_REGISTRY)}")

        if hasattr(self, "dataset"):
            width = width or getattr(self.dataset, "crop_w", None)
            height = height or getattr(self.dataset, "crop_h", None)
            channels = channels or getattr(self.dataset, "channels", None)
        width = width or 32
        height = height or 32
        channels = channels or 1

        if arch in ["n2n", "vit", "swin"]:
            assert width == height, f"{arch} Denoiser currently assumes square crops"
            model_kwargs.setdefault("img_size", width)
        model_kwargs.setdefault("channels", channels)

        self.arch = arch
        self.model = MODEL_REGISTRY[arch](**model_kwargs).to(self.device)

        self.loss_fn = {
            "mse": nn.MSELoss(reduction='none'),
            "mae": nn.L1Loss(reduction='none')}[loss] if isinstance(loss, str) else loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"{self.name} ({arch}) built with {n_params:,} parameters.")
        if summary:
            print(self.model)

    @staticmethod
    def _to_nchw(x):
        # (N, H, W) or (N, H, W, C) to (N, C, H, W)
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        x = x.float()
        if x.ndim == 3:
            x = x.unsqueeze(1) # (N, H, W) -> (N, 1, H, W)
        elif x.ndim == 4:
            x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        return x

    def _make_loader(self, X, Y, batch_size, shuffle, augment=False, MX=None, MY=None):
        X = self._to_nchw(X)
        Y = self._to_nchw(Y)

        has_masks = (MX is not None) and (MY is not None)
        if has_masks:
            MX = self._to_nchw(MX).float()
            MY = self._to_nchw(MY).float()
        else:
            MX = torch.ones_like(X).float()
            MY = torch.ones_like(X).float()

        if augment:
            ds = AFMPairDataset(X, Y, mx_cuts=MX, my_cuts=MY, augment=True)
        else: ds = torch.utils.data.TensorDataset(X, Y, MX, MY)
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    
    def _checkpoint_payload(self):
        return {
            "model_state_dict": self.model.state_dict(),
            "norm_x_min": self.dataset.norm_x_min if self.norm else 0.,
            "norm_scale": self.dataset.norm_scale if self.norm else 1.,
            "use_affine": getattr(self, "use_affine", None),
        }

    def load_model(self, path, map_location=None):
        map_location = map_location or self.device
        checkpoint = torch.load(path, map_location=map_location)

        if "model_state_dict" not in checkpoint:
            self.model.load_state_dict(checkpoint)
            print("Warning: no norm stats found in checkpoint, you must supply them manually.")
            return None

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.norm_x_min = checkpoint["norm_x_min"]
        self.norm_scale = checkpoint["norm_scale"]

        norm = (self.norm_x_min, self.norm_scale)
        print(f"Loaded checkpoint from {path}")
        print(f"-> norm_x_min: {self.norm_x_min}")
        print(f"-> norm_scale: {self.norm_scale}")

    def train(
            self, epochs=50, batch_size=32, v=1, best_validation=True,
            patience=None, augment=False, monitor="val_loss", output="."):
        assert monitor in ("val_loss", "train_val_loss", "train_loss"), \
            "monitor must be 'val_loss' or 'train_val_loss'"

        if not os.path.exists(output):
            os.makedirs(output, exist_ok=True)
        best_model_path = f"{output}/{self.name}.pt"
        history_path = f"{output}/history.pt"

        train_loader = self._make_loader(
            self.train_X, self.train_y, batch_size, shuffle=True,
            augment=augment, MX=self.train_MX, MY=self.train_My)
        train_val_loader = self._make_loader(
            self.train_val_X, self.train_val_y, batch_size, shuffle=False,
            augment=False, MX=self.train_val_MX, MY=self.train_val_My)
        val_loader = self._make_loader(
            self.val_X, self.val_y, batch_size, shuffle=False,
            augment=False, MX=self.val_MX, MY=self.val_My)

        history = {"loss": [], "train_val_loss": [], "val_loss": []}
        best_monitored_loss = float("inf")
        epochs_no_improve = 0

        def _eval(loader):
            self.model.eval()
            running, n_seen = 0.0, 0
            with torch.no_grad():
                for xb, yb, mxb, myb in loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    mxb, myb = mxb.to(self.device), myb.to(self.device)
                    pred = self.model(xb)
                    loss = (self.loss_fn(pred, yb)*(1 - myb)).mean()
                    running += loss.item() * xb.size(0)
                    n_seen += xb.size(0)
            return running / max(n_seen, 1)

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            n_seen = 0
            n_batches = len(train_loader)
            bar_len = 30

            for batch_idx, (xb, yb, mxb, myb) in enumerate(train_loader):
                xb, yb = xb.to(self.device), yb.to(self.device)
                mxb, myb = mxb.to(self.device), myb.to(self.device)

                self.optimizer.zero_grad()
                pred = self.model(xb)
                loss = (self.loss_fn(pred, yb)*(1 - myb)).mean()
                loss.backward()
                self.optimizer.step()

                batch_loss = loss.item()
                running_loss += batch_loss*xb.size(0)
                n_seen += xb.size(0)

                if v > 1:
                    frac = (batch_idx + 1)/n_batches
                    filled = int(bar_len*frac)
                    bar = "#"*filled + "-"*(bar_len - filled)
                    print(f"\rEpoch {epoch + 1}/{epochs} [{bar}] "
                        f"batch {batch_idx + 1}/{n_batches} "
                        f"- batch_loss: {batch_loss:.6f}", end="", flush=True)

            train_loss = running_loss/max(n_seen, 1)
            train_val_loss = _eval(train_val_loader)
            val_loss = _eval(val_loader)

            history["loss"].append(train_loss)
            history["train_val_loss"].append(train_val_loss)
            history["val_loss"].append(val_loss)

            monitored = None
            if monitor == "train_val_loss": monitored = train_val_loss
            if monitor == "val_loss":       monitored = val_loss
            if monitor == "train_loss":     monitored = train_loss

            if monitored < best_monitored_loss:
                best_monitored_loss = monitored
                epochs_no_improve = 0
                torch.save(self._checkpoint_payload(), best_model_path)
            else: epochs_no_improve += 1

            if v:
                epoch_frac = (epoch + 1)/epochs
                epoch_filled = int(bar_len*epoch_frac)
                epoch_bar = "#"*epoch_filled + "-"*(bar_len - epoch_filled)
                add_best = " (*)" if not epochs_no_improve else ""
                print(
                    f"\rEpoch {epoch + 1}/{epochs} [{epoch_bar}] loss: {train_loss:.6f} "
                    f"- train_val_loss: {train_val_loss:.6f} "
                    f"- val_loss: {val_loss:.6f}{add_best}" + " "*15)

            if patience is not None and epochs_no_improve >= patience:
                print(f"Early stopping: no improvement in {monitor} for "
                    f"{patience} epochs (best {monitor}: {best_monitored_loss:.6f})")
                break
        self.history = history
        torch.save(history, history_path)

        if best_validation and os.path.exists(best_model_path):
            self.load_model(best_model_path)

        plt.plot(history["loss"], color="blue", label="loss")
        plt.plot(history["train_val_loss"], color="green", label="train_val_loss")
        plt.plot(history["val_loss"], color="orange", label="val_loss")
        plt.yscale("log")
        plt.legend()
        plt.show()

    def predict(self, x):
        self.model.eval()
        x_t = self._to_nchw(x).to(self.device)
        with torch.no_grad():
            pred = self.model(x_t)
        self.yt = pred.permute(0, 2, 3, 1).cpu().numpy()
        return self.yt

    def fft(self, img):
        F_ = np.fft.fft2(img)
        return np.fft.fftshift(F_)

    def ifft(self, img):
        G = np.fft.ifftshift(img)
        return np.fft.ifft2(G).real

    def fft_mask(self, img, d, apix):
        M, N = img.shape
        nyquist = N / 2
        D0 = nyquist * apix / d
        uu, vv = np.meshgrid(np.arange(M), np.arange(N), indexing="ij")
        D = np.sqrt((uu - M / 2) ** 2 + (vv - N / 2) ** 2)
        return (D <= D0).astype(np.float32)

    def low_pass(self, image, d, apix):
        img = np.asarray(image).copy()
        if img.ndim > 2:
            img = img.squeeze()
        F_ = self.fft(img)
        mask = self.fft_mask(img, d, apix)
        return self.ifft(F_ * mask)
    
    def _np(self, x):
        if torch.is_tensor(x):
            return x.cpu().numpy()
        return np.asarray(x)
    
    def im_compare(self, sample_num=5, columns=4, d=15, apix=5, length=12, height=0.85):

        nums_train = random.sample(range(len(self.train_ids)), sample_num)
        nums_val   = random.sample(range(len(self.train_val_ids)), sample_num)
        nums_test  = random.sample(range(len(self.test_ids)), sample_num)

        train_sample_X = self._np(self.train_X)[nums_train]
        val_sample_X   = self._np(self.val_X)[nums_val]
        test_sample_X  = self._np(self.test_X)[nums_test]
        train_sample_y = self._np(self.train_y)[nums_train]
        val_sample_y   = self._np(self.val_y)[nums_val]
        test_sample_y  = self._np(self.test_y)[nums_test]

        predict_train = self.predict(train_sample_X)
        predict_val   = self.predict(val_sample_X)
        predict_test  = self.predict(test_sample_X)

        new_line = "\n"
        sets = [
            ("Train", nums_train, train_sample_X, train_sample_y, predict_train),
            #("Val", nums_val, val_sample_X, val_sample_y, predict_val),
            #("Test", nums_test, test_sample_X, test_sample_y, predict_test),
        ]

        for label, nums, sample_X, sample_y, predict in sets:
            for i in range(sample_num):
                fig, ax = plt.subplots(1, columns, figsize=(length, height))
                ax[0].imshow(sample_X[i], origin="lower"); ax[0].axis('off')
                ax[0].set_title(f"{label} x {nums[i]}")
                ax[1].imshow(sample_y[i], origin="lower"); ax[1].axis('off')
                ax[1].set_title(f"{label} y {nums[i]}")
                ax[2].imshow(predict[i], origin="lower"); ax[2].axis('off')
                ax[2].set_title(f"Predict {label} y {new_line} from {label} x")
                ax[3].imshow(self.low_pass(sample_X[i], d, apix), origin="lower"); ax[3].axis('off')
                ax[3].set_title(f"Low Pass {new_line} {label} y")
                plt.show()
                plt.close()

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2*window_size - 1)*(2*window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2*window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q@k.transpose(-2, -1))*self.scale

        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(N, N, -1).permute(2, 0, 1).contiguous()
        attn = attn + bias.unsqueeze(0)

        if attn_mask is not None:
            nW = attn_mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out = self.proj(out)
        return out

class SwinBlock(nn.Module):
    def __init__(
            self, dim, input_resolution, num_heads, window_size=4,
            shift_size=0, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        H, W = input_resolution
        if min(H, W) <= window_size:
            shift_size = 0
            window_size = min(H, W)
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim*mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
            w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, window_size).view(-1, window_size*window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            self.register_buffer("attn_mask", attn_mask)
        else:
            self.attn_mask = None

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        windows = window_partition(x, self.window_size).view(-1, self.window_size*self.window_size, C)
        attn_windows = self.attn(windows, attn_mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x.view(B, H*W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

class SwinStage(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=4, mlp_ratio=4.0):
        super().__init__()
        self.block1 = SwinBlock(dim, input_resolution, num_heads, window_size, shift_size=0, mlp_ratio=mlp_ratio)
        self.block2 = SwinBlock(dim, input_resolution, num_heads, window_size, shift_size=window_size // 2, mlp_ratio=mlp_ratio)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x

class SwinDenoiser(nn.Module):
    def __init__(
            self, img_size=32, channels=1, patch_size=4, dim=192,
            depth=6, num_heads=6, window_size=4, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.channels = channels
        self.grid = img_size // patch_size
        num_patches = self.grid**2
        patch_dim = channels*patch_size*patch_size

        assert self.grid % window_size == 0 or window_size >= self.grid, \
            f"window_size ({window_size}) should divide grid ({self.grid}) evenly"

        self.patch_embed = nn.Linear(patch_dim, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        n_stages = max(1, depth // 2)
        self.stages = nn.ModuleList([
            SwinStage(dim, (self.grid, self.grid), num_heads, window_size, mlp_ratio)
            for _ in range(n_stages)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, patch_dim)

    def _patchify(self, x):
        b, c, h, w = x.shape
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(b, self.grid*self.grid, c*p*p)
        return x

    def _unpatchify(self, tokens, batch_size):
        p = self.patch_size
        c = self.channels
        g = self.grid
        x = tokens.view(batch_size, g, g, c, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(batch_size, c, g*p, g*p)
        return x

    def forward(self, x):
        b, c, h, w = x.shape
        if h != self.img_size or w != self.img_size:
            raise ValueError(
                f"SwinDenoiser built with img_size={self.img_size} but received {h}x{w}")
        tokens = self.patch_embed(self._patchify(x)) + self.pos_embed
        for stage in self.stages:
            tokens = stage(tokens)
        tokens = self.norm(tokens)
        patches_out = self.head(tokens)
        return self._unpatchify(patches_out, b)

MODEL_REGISTRY = {
    "n2n": N2N,
    "unet": UNet,
    "vit": ViTDenoiser,
    "attn_unet": AttnUNet,
    "swin": SwinDenoiser
}
