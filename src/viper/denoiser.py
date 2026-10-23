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
        """
        PyTorch Dataset pairing two-direction AFM image crops,
        with optional masks and optional dihedral-group data augmentation.

        Parameters
        ----------
        x_cuts : torch.Tensor
            Input image crops, shape (N, C, H, W).
        y_cuts : torch.Tensor
            Target image crops, shape (N, C, H, W), paired with `x_cuts`.
        mx_cuts : torch.Tensor or None, optional
            Mask crops corresponding to `x_cuts`, by default None.
        my_cuts : torch.Tensor or None, optional
            Mask crops corresponding to `y_cuts`, by default None.
        augment : bool, optional
            If True, apply a randomly chosen flip/rotation to each sample (and
            its masks, if present) on retrieval, by default True.
        """
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
        """
        Returns
        -------
        int
            Number of samples in the dataset.
        """
        return self.x_cuts.shape[0]

    @staticmethod
    def _apply_transform(img, op):
        """
        Apply one of a fixed set of dihedral transforms to an image tensor.

        Parameters
        ----------
        img : torch.Tensor
            Image tensor of shape (C, H, W).
        op : int
            Transform code: 0 for identity, 1 for horizontal flip, 2 for
            vertical flip, 3 for 180-degree rotation.

        Returns
        -------
        torch.Tensor
            Transformed image tensor of the same shape as `img`.
        """
        # img shape is (C, H, W)
        if op == 1: # horizontal flip (width)
            return torch.flip(img, dims=(-1,))           
        if op == 2: # vertical flip (height)
            return torch.flip(img, dims=(-2,))           
        if op == 3: # 180 degree rotation
            return torch.rot90(img, k=2, dims=(-2, -1))  
        return img  # identity

    def __getitem__(self, idx):
        """
        Retrieve a (possibly augmented) sample from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple of torch.Tensor
            `(x, y)` if no masks are present, otherwise `(x, y, mx, my)`,
            with the same random transform applied to all tensors in the
            tuple when augmentation is enabled.
        """
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
        """
        A small convolutional block of one or two Conv2d + activation layers
        with "same" padding.

        Parameters
        ----------
        in_ch : int
            Number of input channels.
        out_ch : int
            Number of output channels.
        kernel_size : int, optional
            Convolution kernel size, by default 3.
        activation : {"relu", "leaky_relu"}, optional
            Activation function to use, by default "relu".
        double : bool, optional
            If True, apply a second Conv2d + activation (out_ch -> out_ch)
            after the first, by default False.
        """
        super().__init__()
        act = {"relu": nn.ReLU, "leaky_relu": lambda: nn.LeakyReLU(0.01)}[activation]
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size, padding="same"), act()]
        if double:
            layers += [nn.Conv2d(out_ch, out_ch, kernel_size, padding="same"), act()]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, in_ch, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, out_ch, H, W).
        """
        return self.block(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, activation="relu", kernel_size=3, stride=2):
        """
        U-Net style decoder block: transposed-convolution upsampling followed
        by concatenation with a skip connection and a double convolution.

        Parameters
        ----------
        in_ch : int
            Number of input channels (from the previous decoder stage or
            bottleneck).
        skip_ch : int
            Number of channels in the skip connection to concatenate.
        out_ch : int
            Number of output channels after upsampling and after the final
            convolution.
        activation : {"relu", "leaky_relu"}, optional
            Activation function to use, by default "relu".
        kernel_size : int, optional
            Kernel size of the transposed convolution, by default 3.
        stride : int, optional
            Stride (upsampling factor) of the transposed convolution, by
            default 2.
        """
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=kernel_size, stride=stride,
            padding=kernel_size // 2, output_padding=stride - 1)
        self.conv = ConvBlock(
            out_ch + skip_ch, out_ch, activation=activation, double=True)

    def forward(self, x, skip):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input feature map to be upsampled, shape (B, in_ch, H, W).
        skip : torch.Tensor
            Skip-connection feature map to concatenate after upsampling,
            shape (B, skip_ch, H', W').

        Returns
        -------
        torch.Tensor
            Output feature map of shape (B, out_ch, H', W'). If the
            upsampled size doesn't match the skip connection's spatial
            size, it is resized via nearest-neighbor interpolation first.
        """
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class SelfAttention2d(nn.Module):
    def __init__(self, channels, num_heads=4):
        """
        Multi-head self-attention over the spatial positions of a feature map,
        applied as a residual block with GroupNorm pre-normalization.

        Parameters
        ----------
        channels : int
            Number of input/output channels.
        num_heads : int, optional
            Number of attention heads, by default 4.
        """
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input feature map of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Output feature map of shape (B, C, H, W), equal to the input
            plus the self-attention residual.
        """
        b, c, h, w = x.shape
        residual = x
        x = self.norm(x)
        tokens = x.flatten(2).transpose(1, 2)
        attended, _ = self.attn(tokens, tokens, tokens)
        attended = attended.transpose(1, 2).reshape(b, c, h, w)
        return residual + attended

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.0):
        """
        Standard pre-norm Transformer encoder block with multi-head
        self-attention and an MLP, each with a residual connection.

        Parameters
        ----------
        dim : int
            Token embedding dimension.
        num_heads : int, optional
            Number of attention heads, by default 8.
        mlp_ratio : float, optional
            Ratio of the MLP hidden dimension to `dim`, by default 4.0.
        dropout : float, optional
            Dropout probability applied in attention and the MLP, by default
            0.0.
        """
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
        """
        Parameters
        ----------
        x : torch.Tensor
            Input token sequence of shape (B, N, dim).

        Returns
        -------
        torch.Tensor
            Output token sequence of shape (B, N, dim).
        """
        a, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x

class N2N(nn.Module):
    def __init__(self, channels=1, base_filters=48, depth=5, activation="leaky_relu", img_size=32):
        """
        A Noise2Noise-style encoder-decoder convolutional denoiser with
        single-conv encoder stages, a convolutional bottleneck, and
        transposed-convolution decoder stages with skip connections (including
        the raw input as the final skip).

        Parameters
        ----------
        channels : int, optional
            Number of input/output image channels, by default 1.
        base_filters : int, optional
            Number of filters used in the stem and encoder/bottleneck
            convolutions, by default 48.
        depth : int, optional
            Number of encoder/decoder stages requested. Automatically reduced
            if it exceeds what `img_size` can support, by default 5.
        activation : {"relu", "leaky_relu"}, optional
            Activation function used throughout, by default "leaky_relu".
        img_size : int, optional
            Expected input image size (height/width), used only to cap
            `depth`, by default 32.
        """
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
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (B, channels, H, W).

        Returns
        -------
        torch.Tensor
            Denoised output tensor of shape (B, channels, H, W), resized to
            match the input spatial size if needed.
        """
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
        """
        A standard U-Net with double-convolution encoder/decoder stages,
        max-pooling downsampling, a convolutional bottleneck, and
        transposed-convolution upsampling with skip connections.

        Parameters
        ----------
        channels : int, optional
            Number of input/output image channels, by default 1.
        filters : tuple of int, optional
            Number of filters at each encoder stage (and, reversed, each
            decoder stage), by default (16, 32, 64, 128, 256).
        bottleneck_filters : int, optional
            Number of filters in the bottleneck convolutions, by default 512.
        activation : {"relu", "leaky_relu"}, optional
            Activation function used throughout, by default "relu".
        """
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
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (B, channels, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, channels, H, W).
        """
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
        """
        A Vision Transformer-based image denoiser: splits the image into
        non-overlapping patches, processes them with a stack of Transformer
        blocks, and linearly projects tokens back to pixel patches.

        Parameters
        ----------
        img_size : int, optional
            Expected input image height and width (must be square), by default
            32.
        channels : int, optional
            Number of input/output image channels, by default 1.
        patch_size : int, optional
            Side length of each square patch; must evenly divide `img_size`,
            by default 4.
        dim : int, optional
            Token embedding dimension, by default 192.
        depth : int, optional
            Number of Transformer blocks, by default 6.
        num_heads : int, optional
            Number of attention heads per block, by default 6.
        mlp_ratio : float, optional
            Ratio of MLP hidden dimension to `dim`, by default 4.0.
        dropout : float, optional
            Dropout probability, by default 0.0.
        """
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
        """
        Split an image batch into flattened non-overlapping patches.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Patch tokens of shape (B, grid*grid, C*patch_size*patch_size).
        """
        b, c, h, w = x.shape
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p) # (B, C, gh, gw, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(b, self.grid * self.grid, c * p * p)
        return x

    def _unpatchify(self, tokens, batch_size):
        """
        Reassemble flattened patch tokens back into an image batch.

        Parameters
        ----------
        tokens : torch.Tensor
            Patch tokens of shape (B, grid*grid, C*patch_size*patch_size).
        batch_size : int
            Batch size, used to reshape the tokens.

        Returns
        -------
        torch.Tensor
            Reconstructed image tensor of shape (B, C, H, W).
        """
        p = self.patch_size
        c = self.channels
        g = self.grid
        x = tokens.view(batch_size, g, g, c, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(batch_size, c, g * p, g * p)
        return x

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (B, channels, img_size, img_size).

        Returns
        -------
        torch.Tensor
            Output tensor of the same shape as the input.

        Raises
        ------
        ValueError
            If the input spatial dimensions don't match `img_size`.
        """
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
        """
        A U-Net variant with self-attention blocks inserted at deeper
        encoder/decoder stages and in the bottleneck.

        Parameters
        ----------
        channels : int, optional
            Number of input/output image channels, by default 1.
        filters : tuple of int, optional
            Number of filters at each encoder stage (and, reversed, each
            decoder stage), by default (16, 32, 64, 128, 256).
        bottleneck_filters : int, optional
            Number of filters in the bottleneck convolution, by default 512.
        activation : {"relu", "leaky_relu"}, optional
            Activation function used throughout, by default "relu".
        attn_heads : int, optional
            Number of attention heads used in each `SelfAttention2d` block, by
            default 4.
        attn_from_stage : int, optional
            Index of the shallowest encoder stage (0-indexed) at which
            self-attention is applied; attention is also applied to decoder
            stages at an equivalent depth from the bottleneck, by default 2.
        """
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
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (B, channels, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, channels, H, W).
        """
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
        """
        High-level training/evaluation wrapper around the denoiser
        architectures in `MODEL_REGISTRY`, handling dataset splitting, model
        construction, training with masked loss, checkpointing, prediction, and
        visualization.

        Parameters
        ----------
        name : str, optional
            Name used for logging and checkpoint filenames, by default
            "Denoiser".
        device : str, optional
            Torch device string (e.g. "cpu" or "cuda") to run the model on, by
            default "cpu".
        """
        self.name = name
        self.device = torch.device(device)
        self.model = None

    def set_dataset(self, data):
        """
        Attach a dataset object providing paired AFM crops to the wrapper.

        Parameters
        ----------
        data : object
            Dataset object exposing `cuts_num`, `x_cuts`/`x_norm`,
            `y_cuts`/`y_norm`, `mx_cuts`, `my_cuts`, and optionally
            `crop_w`, `crop_h`, `channels`.

        Returns
        -------
        None
        """
        self.dataset = data

    def split(self, fractions=(0.7, 0.2, 0.1), norm=False):
        """
        Randomly split the attached dataset into train, train-validation,
        and test subsets, and construct a reversed-direction (Y -> X)
        validation set from the train-validation crops.

        Parameters
        ----------
        fractions : tuple of float, optional
            Approximate (train, train_val, test) fractions of the dataset,
            by default (0.7, 0.2, 0.1). The test fraction is implied by the
            remainder.
        norm : bool, optional
            If True, use the dataset's normalized arrays (`x_norm`,
            `y_norm`) instead of the raw arrays, by default False.

        Returns
        -------
        None

        Notes
        -----
        Populates `self.train_X/y/MX/My`, `self.train_val_X/y/MX/My`,
        `self.val_X/y/MX/My` (built by swapping X and Y on the
        train-validation crops), and `self.test_X/y/MX/My`, along with the
        corresponding id lists.
        """
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
        """
        Construct the underlying denoiser model, loss function, and
        optimizer.

        Parameters
        ----------
        arch : str, optional
            Architecture name, one of the keys in `MODEL_REGISTRY`
            ("n2n", "unet", "vit", "attn_unet", "swin"), by default "unet".
        width : int or None, optional
            Expected crop width. If None, inferred from the attached
            dataset's `crop_w`, falling back to 32, by default None.
        height : int or None, optional
            Expected crop height. If None, inferred from the attached
            dataset's `crop_h`, falling back to 32, by default None.
        channels : int or None, optional
            Number of image channels. If None, inferred from the attached
            dataset's `channels`, falling back to 1, by default None.
        lr : float, optional
            Learning rate for the Adam optimizer, by default 1e-3.
        loss : {"mse", "mae"} or callable, optional
            Loss function to use (elementwise, reduction="none"), or a
            custom callable, by default "mse".
        summary : bool, optional
            If True, print the full model architecture, by default False.
        **model_kwargs
            Additional keyword arguments forwarded to the model
            constructor.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `arch` is not a key in `MODEL_REGISTRY`.
        AssertionError
            If `arch` requires square crops (n2n, vit, swin) and
            `width != height`.
        """
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
        """
        Convert an array/tensor in (N, H, W) or (N, H, W, C) layout to
        (N, C, H, W) float tensor layout.

        Parameters
        ----------
        x : array_like or torch.Tensor
            Input batch of images.

        Returns
        -------
        torch.Tensor
            Float tensor of shape (N, C, H, W).
        """
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        x = x.float()
        if x.ndim == 3:
            x = x.unsqueeze(1) # (N, H, W) -> (N, 1, H, W)
        elif x.ndim == 4:
            x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        return x

    def _make_loader(self, X, Y, batch_size, shuffle, augment=False, MX=None, MY=None):
        """
        Build a DataLoader over paired input/target images and masks.

        Parameters
        ----------
        X : array_like or torch.Tensor
            Input images.
        Y : array_like or torch.Tensor
            Target images.
        batch_size : int
            Batch size for the DataLoader.
        shuffle : bool
            Whether to shuffle samples each epoch.
        augment : bool, optional
            If True, wrap the data in an `AFMPairDataset` with random
            flip/rotation augmentation, by default False.
        MX : array_like or torch.Tensor or None, optional
            Mask corresponding to `X`. If None (together with `MY`), an
            all-ones mask is used, by default None.
        MY : array_like or torch.Tensor or None, optional
            Mask corresponding to `Y`. If None (together with `MX`), an
            all-ones mask is used, by default None.

        Returns
        -------
        torch.utils.data.DataLoader
            DataLoader yielding `(x, y, mx, my)` batches.
        """
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
        """
        Assemble the dictionary saved to a model checkpoint.

        Returns
        -------
        dict
            Dictionary with keys "model_state_dict", "norm_x_min",
            "norm_scale", and "use_affine", reflecting the current model
            and (if `self.norm` is True) the dataset's normalization
            parameters.
        """
        return {
            "model_state_dict": self.model.state_dict(),
            "norm_x_min": self.dataset.norm_x_min if self.norm else 0.,
            "norm_scale": self.dataset.norm_scale if self.norm else 1.,
            "use_affine": getattr(self, "use_affine", None),
        }

    def load_model(self, path, map_location=None):
        """
        Load model weights (and, if present, normalization statistics)
        from a checkpoint file.

        Parameters
        ----------
        path : str
            Path to the checkpoint file.
        map_location : str or torch.device or None, optional
            Device mapping passed to `torch.load`. Defaults to
            `self.device` if None, by default None.

        Returns
        -------
        None

        Notes
        -----
        If the checkpoint does not contain a "model_state_dict" key, it is
        assumed to be a raw state dict, which is loaded directly; a warning
        is printed that normalization statistics must be supplied manually.
        """
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
        """
        Train the model with a masked reconstruction loss, tracking
        training, same-direction validation, and reversed-direction
        validation losses, with optional early stopping and checkpointing.

        Parameters
        ----------
        epochs : int, optional
            Maximum number of training epochs, by default 50.
        batch_size : int, optional
            Batch size used for training and evaluation, by default 32.
        v : int, optional
            Verbosity level: 0 for silent, 1 for per-epoch progress, >1 for
            per-batch progress bars, by default 1.
        best_validation : bool, optional
            If True, reload the best checkpoint (by `monitor`) at the end
            of training, by default True.
        patience : int or None, optional
            Number of epochs without improvement in `monitor` after which
            training stops early. If None, no early stopping is applied, by
            default None.
        augment : bool, optional
            If True, apply random flip/rotation augmentation to the
            training set, by default False.
        monitor : {"val_loss", "train_val_loss", "train_loss"}, optional
            Metric used to select the best checkpoint and for early
            stopping, by default "val_loss".
        output : str, optional
            Directory in which to save the best model checkpoint and
            training history, by default ".".

        Returns
        -------
        None

        Notes
        -----
        The masked loss is computed as `loss_fn(pred, target) * (1 - mask)`,
        averaged over all elements, so masked-out regions (mask == 1) are
        excluded. Saves `{output}/{self.name}.pt` (best checkpoint) and
        `{output}/history.pt` (loss history), and displays a plot of the
        training/validation loss curves on a log scale.
        """
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
        """
        Run the model on a batch of input images.

        Parameters
        ----------
        x : array_like or torch.Tensor
            Input images in (N, H, W) or (N, H, W, C) layout.

        Returns
        -------
        numpy.ndarray
            Predicted images in (N, H, W, C) layout, also cached as
            `self.yt`.
        """
        self.model.eval()
        x_t = self._to_nchw(x).to(self.device)
        with torch.no_grad():
            pred = self.model(x_t)
        self.yt = pred.permute(0, 2, 3, 1).cpu().numpy()
        return self.yt

    def fft(self, img):
        """
        Compute the centered 2D FFT of an image.

        Parameters
        ----------
        img : numpy.ndarray
            2D input image.

        Returns
        -------
        numpy.ndarray
            Complex-valued, zero-frequency-centered 2D FFT of `img`.
        """
        F_ = np.fft.fft2(img)
        return np.fft.fftshift(F_)

    def ifft(self, img):
        """
        Compute the real part of the inverse 2D FFT of a centered spectrum.

        Parameters
        ----------
        img : numpy.ndarray
            Zero-frequency-centered 2D Fourier spectrum.

        Returns
        -------
        numpy.ndarray
            Real-valued reconstructed 2D image.
        """
        G = np.fft.ifftshift(img)
        return np.fft.ifft2(G).real

    def fft_mask(self, img, d, apix):
        """
        Build a circular low-pass binary mask in frequency space
        corresponding to a real-space resolution cutoff.

        Parameters
        ----------
        img : numpy.ndarray
            2D reference image used only for its shape.
        d : float
            Desired real-space resolution cutoff (same length units as
            `apix`).
        apix : float
            Pixel size (length units per pixel).

        Returns
        -------
        numpy.ndarray
            Binary mask of the same shape as `img`, 1 inside the cutoff
            radius and 0 outside.
        """
        M, N = img.shape
        nyquist = N / 2
        D0 = nyquist * apix / d
        uu, vv = np.meshgrid(np.arange(M), np.arange(N), indexing="ij")
        D = np.sqrt((uu - M / 2) ** 2 + (vv - N / 2) ** 2)
        return (D <= D0).astype(np.float32)

    def low_pass(self, image, d, apix):
        """
        Apply a circular low-pass filter to an image in Fourier space.

        Parameters
        ----------
        image : array_like
            Input image, optionally with a singleton channel dimension.
        d : float
            Desired real-space resolution cutoff.
        apix : float
            Pixel size (length units per pixel).

        Returns
        -------
        numpy.ndarray
            Low-pass filtered 2D image.
        """
        img = np.asarray(image).copy()
        if img.ndim > 2:
            img = img.squeeze()
        F_ = self.fft(img)
        mask = self.fft_mask(img, d, apix)
        return self.ifft(F_ * mask)
    
    def _np(self, x):
        """
        Convert a tensor or array-like object to a NumPy array.

        Parameters
        ----------
        x : torch.Tensor or array_like
            Input data.

        Returns
        -------
        numpy.ndarray
            NumPy array view/copy of the input.
        """
        if torch.is_tensor(x):
            return x.cpu().numpy()
        return np.asarray(x)
    
    def im_compare(self, sample_num=5, columns=4, d=15, apix=5, length=12, height=0.85):
        """
        Visually compare random training samples against model predictions
        and a low-pass-filtered reference.

        Parameters
        ----------
        sample_num : int, optional
            Number of random training samples to display, by default 5.
        columns : int, optional
            Number of subplot columns per sample (input, target, prediction,
            low-pass reference), by default 4.
        d : float, optional
            Resolution cutoff passed to `low_pass`, by default 15.
        apix : float, optional
            Pixel size passed to `low_pass`, by default 5.
        length : float, optional
            Figure width in inches, by default 12.
        height : float, optional
            Figure height in inches, by default 0.85.

        Returns
        -------
        None

        Notes
        -----
        Only the "Train" set is currently displayed; validation and test
        comparisons are computed but not plotted.
        """
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
    """
    Partition a batch of feature maps into non-overlapping square windows.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (B, H, W, C).
    window_size : int
        Side length of each square window; must evenly divide H and W.

    Returns
    -------
    torch.Tensor
        Windows of shape (B * H/window_size * W/window_size, window_size,
        window_size, C).
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Reassemble windowed feature maps back into a full batch of feature
    maps.

    Parameters
    ----------
    windows : torch.Tensor
        Windowed tensor of shape (B * num_windows, window_size,
        window_size, C), as produced by `window_partition`.
    window_size : int
        Side length of each square window.
    H : int
        Full feature map height.
    W : int
        Full feature map width.

    Returns
    -------
    torch.Tensor
        Reassembled tensor of shape (B, H, W, C).
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, dropout=0.0):
        """
        Multi-head self-attention restricted to a local window, with a learned
        relative positional bias (as in Swin Transformer).

        Parameters
        ----------
        dim : int
            Token embedding dimension.
        window_size : int
            Side length of the square attention window.
        num_heads : int
            Number of attention heads.
        qkv_bias : bool, optional
            Whether to include a bias term in the QKV projection, by default
            True.
        dropout : float, optional
            Dropout probability applied to attention weights, by default 0.0.
        """
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
        """
        Parameters
        ----------
        x : torch.Tensor
            Input window tokens of shape (B_, N, C), where B_ is the number
            of windows (across the batch) and N = window_size**2.
        attn_mask : torch.Tensor or None, optional
            Additive attention mask of shape (num_windows, N, N) used to
            prevent attention across shifted-window boundaries, by default
            None.

        Returns
        -------
        torch.Tensor
            Output window tokens of shape (B_, N, C).
        """
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
        """
        A single Swin Transformer block: windowed (optionally shifted)
        self-attention followed by an MLP, each with a residual connection and
        pre-normalization.

        Parameters
        ----------
        dim : int
            Token embedding dimension.
        input_resolution : tuple of int
            Spatial resolution (H, W) of the token grid at this stage.
        num_heads : int
            Number of attention heads.
        window_size : int, optional
            Side length of the attention window. Reduced to `min(H, W)` (with
            shifting disabled) if the input resolution is too small, by default
            4.
        shift_size : int, optional
            Number of positions to cyclically shift the token grid before
            windowing, enabling cross-window connections; 0 for no shift, by
            default 0.
        mlp_ratio : float, optional
            Ratio of MLP hidden dimension to `dim`, by default 4.0.
        dropout : float, optional
            Dropout probability, by default 0.0.
        """
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
        """
        Parameters
        ----------
        x : torch.Tensor
            Input token sequence of shape (B, H*W, dim).

        Returns
        -------
        torch.Tensor
            Output token sequence of shape (B, H*W, dim).
        """
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
        """
        A pair of consecutive Swin blocks: one with regular window attention
        and one with shifted window attention.

        Parameters
        ----------
        dim : int
            Token embedding dimension.
        input_resolution : tuple of int
            Spatial resolution (H, W) of the token grid at this stage.
        num_heads : int
            Number of attention heads.
        window_size : int, optional
            Side length of the attention window, by default 4.
        mlp_ratio : float, optional
            Ratio of MLP hidden dimension to `dim`, by default 4.0.
        """
        super().__init__()
        self.block1 = SwinBlock(dim, input_resolution, num_heads, window_size, shift_size=0, mlp_ratio=mlp_ratio)
        self.block2 = SwinBlock(dim, input_resolution, num_heads, window_size, shift_size=window_size // 2, mlp_ratio=mlp_ratio)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input token sequence of shape (B, H*W, dim).

        Returns
        -------
        torch.Tensor
            Output token sequence of shape (B, H*W, dim).
        """
        x = self.block1(x)
        x = self.block2(x)
        return x

class SwinDenoiser(nn.Module):
    def __init__(
            self, img_size=32, channels=1, patch_size=4, dim=192,
            depth=6, num_heads=6, window_size=4, mlp_ratio=4.0, dropout=0.0):
        """
        A Swin Transformer-based image denoiser: splits the image into patches,
        processes them through a stack of `SwinStage` blocks with windowed
        attention, and linearly projects tokens back to pixel patches.

        Parameters
        ----------
        img_size : int, optional
            Expected input image height and width (must be square), by default
            32.
        channels : int, optional
            Number of input/output image channels, by default 1.
        patch_size : int, optional
            Side length of each square patch; must evenly divide `img_size`, by
            default 4.
        dim : int, optional
            Token embedding dimension, by default 192.
        depth : int, optional
            Total number of Swin blocks requested; organized into
            `max(1, depth // 2)` two-block stages, by default 6.
        num_heads : int, optional
            Number of attention heads per block, by default 6.
        window_size : int, optional
            Side length of the attention window; should evenly divide the
            patch grid size (or be at least as large as it), by default 4.
        mlp_ratio : float, optional
            Ratio of MLP hidden dimension to `dim`, by default 4.0.
        dropout : float, optional
            Dropout probability, by default 0.0.
        """
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
        """
        Split an image batch into flattened non-overlapping patches.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Patch tokens of shape (B, grid*grid, C*patch_size*patch_size).
        """
        b, c, h, w = x.shape
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(b, self.grid*self.grid, c*p*p)
        return x

    def _unpatchify(self, tokens, batch_size):
        """
        Reassemble flattened patch tokens back into an image batch.

        Parameters
        ----------
        tokens : torch.Tensor
            Patch tokens of shape (B, grid*grid, C*patch_size*patch_size).
        batch_size : int
            Batch size, used to reshape the tokens.

        Returns
        -------
        torch.Tensor
            Reconstructed image tensor of shape (B, C, H, W).
        """
        p = self.patch_size
        c = self.channels
        g = self.grid
        x = tokens.view(batch_size, g, g, c, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(batch_size, c, g*p, g*p)
        return x

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (B, channels, img_size, img_size).

        Returns
        -------
        torch.Tensor
            Output tensor of the same shape as the input.

        Raises
        ------
        ValueError
            If the input spatial dimensions don't match `img_size`.
        """
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
