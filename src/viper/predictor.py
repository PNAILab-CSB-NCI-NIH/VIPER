import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from denoiser import MODEL_REGISTRY
from afm import AFMImage

class Predictor:
    def __init__(self, arch, crop_size, channels=1, device="cpu", **model_kwargs):
        if arch not in MODEL_REGISTRY:
            raise ValueError(f"Unknown arch '{arch}'. Options: {list(MODEL_REGISTRY)}")

        self.arch = arch
        self.crop_size = crop_size
        self.channels = channels
        self.device = torch.device(device)
        self.output = "."

        if arch in ("vit", "n2n"):
            model_kwargs.setdefault("img_size", crop_size)
        model_kwargs.setdefault("channels", channels)

        self.model = MODEL_REGISTRY[arch](**model_kwargs).to(self.device)
        self.model.eval()
        
    def load(self, path, map_location=None, output=None):
        self.output = output if output else Path(path).parent.name + "/denoised"
        if not os.path.exists(self.output):
            os.makedirs(self.output, exist_ok=True)

        map_location = map_location or self.device
        checkpoint = torch.load(path, map_location=map_location)
        
        if "model_state_dict" not in checkpoint:
            self.model.load_state_dict(checkpoint)
            print("Warning: no norm stats found in checkpoint, you must supply them manually.")
            return None

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.norm_x_min = checkpoint["norm_x_min"]
        self.norm_scale = checkpoint["norm_scale"]

        self.norm = (self.norm_x_min, self.norm_scale)
        print(f"Loaded checkpoint from {path}")
        print(f"-> norm_x_min: {self.norm_x_min}")
        print(f"-> norm_scale: {self.norm_scale}")
    
    @staticmethod
    def _make_weight_map(size, sharpness=8.0):
        coords = np.linspace(-1, 1, size)
        # sigmoid-based bump: 1 at center, ~0 at edges
        bump_1d = 1.0/(1.0 + np.exp(sharpness * (np.abs(coords) - 0.5)))
        bump_1d = bump_1d/bump_1d.max() # normalize peak to 1
        bump_1d = np.clip(bump_1d, 1e-3, 1.0) # floor -> never 0
        weight_2d = np.outer(bump_1d, bump_1d)
        return weight_2d.astype(np.float32)

    def predict_image(self, image, stride=None, batch_size=32, sharpness=8.0):
        if torch.is_tensor(image):
            image = image.cpu().numpy()
        image = np.asarray(image, dtype=np.float32)
        H, W = image.shape
        cs = self.crop_size
        stride = stride or (cs // 2)

        if H < cs or W < cs:
            raise ValueError(f"Image ({H}x{W}) smaller than crop_size ({cs})")

        norm_x_min, norm_scale = self.norm
        norm_x_min = float(norm_x_min.item() if torch.is_tensor(norm_x_min) else norm_x_min)
        norm_scale = float(norm_scale.item() if torch.is_tensor(norm_scale) else norm_scale)
        if norm_scale == 0:
            raise ValueError("norm_scale is 0: cannot normalize")

        # patch top-left coordinates, stepping by stride
        ys = list(range(0, H - cs + 1, stride))
        if ys[-1] != H - cs:
            ys.append(H - cs)
        xs = list(range(0, W - cs + 1, stride))
        if xs[-1] != W - cs:
            xs.append(W - cs)

        weight_map = self._make_weight_map(cs, sharpness=sharpness)
        accum = np.zeros((H, W), dtype=np.float64)
        weight_sum = np.zeros((H, W), dtype=np.float64)

        # gather all patch coords, batch through the model
        coords = [(y, x) for y in ys for x in xs]
        for batch_start in range(0, len(coords), batch_size):
            batch_coords = coords[batch_start: batch_start + batch_size]
            patches = np.stack(
                [image[y:y + cs, x:x + cs] for y, x in batch_coords], axis=0
            )

            # Normalize each patch using the training-set stats
            patches = (patches - norm_x_min)/norm_scale

            patch_tensor = torch.from_numpy(patches).float().unsqueeze(1).to(self.device)
            with torch.no_grad():
                preds = self.model(patch_tensor)
            preds = preds.squeeze(1).cpu().numpy()

            # Map predictions back to original units before accumulating
            preds = preds * norm_scale + norm_x_min
            for (y, x), pred in zip(batch_coords, preds):
                accum[y:y + cs, x:x + cs] += pred*weight_map
                weight_sum[y:y + cs, x:x + cs] += weight_map

        denoised = accum / np.clip(weight_sum, 1e-8, None)
        return denoised.astype(np.float32)

    def _np(self, x):
        if torch.is_tensor(x):
            return x.cpu().numpy()
        return np.asarray(x)

    def compare(
            self, trace, retrace, prediction, cmap="afmhot",
            figsize=(15, 5), pred_max=True, output=None):
        trace      = self._np(trace).squeeze()
        retrace    = self._np(retrace).squeeze()
        prediction = self._np(prediction).squeeze()

        fig, ax = plt.subplots(1, 4, figsize=figsize)
        vmin, vmax = None, None
        if pred_max:
            vmin, vmax = prediction.min(), prediction.max()

        im0 = ax[0].imshow(trace, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax[0].set_title("Trace (input)")
        ax[0].axis("off")
        fig.colorbar(im0, ax=ax[0], fraction=0.05)

        im1 = ax[1].imshow(retrace, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax[1].set_title("Retrace (reference)")
        ax[1].axis("off")
        fig.colorbar(im1, ax=ax[1], fraction=0.05)

        im2 = ax[2].imshow(prediction, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax[2].set_title("Prediction (denoised)")
        ax[2].axis("off")
        fig.colorbar(im2, ax=ax[2], fraction=0.05)

        diff = prediction - trace
        #vmax = np.abs(diff).max()
        #vmin = -vmax
        im3 = ax[3].imshow(diff, origin="lower", cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax[3].set_title("Prediction - Trace")
        ax[3].axis("off")
        fig.colorbar(im3, ax=ax[3], fraction=0.05)

        plt.tight_layout()
        if output:
            if len(output.split("/")) == 1:
                output = self.output + "/" + output
            plt.savefig(output)
        plt.show()

        return fig, ax

    def write_image(self, pred, apix, name):
        if len(name.split("/")) == 1:
            name = self.output + "/" + name
        img = AFMImage()
        img.write_image(pred, apix, name)