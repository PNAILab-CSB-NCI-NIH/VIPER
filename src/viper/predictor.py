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
        """
        Initialize a Predictor with a registered denoising model architecture.

        Parameters
        ----------
        arch : str
            Model architecture name, must be a key in MODEL_REGISTRY.
        crop_size : int
            Spatial size of the square patches the model operates on.
        channels : int, optional
            Number of input channels, by default 1.
        device : str, optional
            Device to run the model on, by default "cpu".
        **model_kwargs
            Additional keyword arguments passed to the model constructor.
            """
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
        """
        Load model weights and normalization statistics from a checkpoint file.

        Parameters
        ----------
        path : str or Path
            Path to the checkpoint file.
        map_location : str or torch.device, optional
            Device mapping for loading the checkpoint. Defaults to self.device.
        output : str, optional
            Output directory for denoised results. Defaults to a "denoised"
            subfolder next to the checkpoint.
        """
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
        """
        Construct a 2D sigmoid-based weight map that peaks at the center and tapers toward the edges.

        Parameters
        ----------
        size : int
            Side length of the square weight map.
        sharpness : float, optional
            Steepness of the sigmoid falloff, by default 8.0.
        
        Returns
        -------
        ndarray
            2D weight map of shape (size, size), normalized to a peak of 1.
        """
        coords = np.linspace(-1, 1, size)
        # sigmoid-based bump: 1 at center, ~0 at edges
        bump_1d = 1.0/(1.0 + np.exp(sharpness * (np.abs(coords) - 0.5)))
        bump_1d = bump_1d/bump_1d.max() # normalize peak to 1
        bump_1d = np.clip(bump_1d, 1e-3, 1.0) # floor -> never 0
        weight_2d = np.outer(bump_1d, bump_1d)
        return weight_2d.astype(np.float32)

    def predict_image(self, image, stride=None, batch_size=32, sharpness=8.0):
        """
        Denoise a full image by predicting over overlapping patches and blending the results.

        Parameters
        ----------
        image : array_like
            Input 2D image to denoise.
        stride : int, optional
            Step size between patches. Defaults to half the crop size.
        batch_size : int, optional
            Number of patches to process per model forward pass, by default 32.
        sharpness : float, optional
            Sharpness of the patch-blending weight map, by default 8.0.

        Returns
        -------
        ndarray
            Denoised image of the same shape as the input.
        """
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

        denoised = accum/np.clip(weight_sum, 1e-8, None)
        return denoised.astype(np.float32)

    def _np(self, x):
        """
        Convert a tensor or array-like input to a NumPy array.

        Parameters
        ----------
        x : torch.Tensor or array_like
            Input data.
        
        Returns
        -------
        ndarray
            NumPy array representation of x.
        """
        if torch.is_tensor(x):
            return x.cpu().numpy()
        return np.asarray(x)

    def compare(
            self, trace, retrace, prediction, cmap="afmhot",
            figsize=(15, 5), pred_max=True, output=None):
        """
        Plot trace, retrace, prediction, and prediction-minus-trace difference side by side.
        
        Parameters
        ----------
        trace : array_like
            Input (trace) image.
        retrace : array_like
            Reference (retrace) image.
        prediction : array_like
            Denoised prediction image.
        cmap : str, optional
            Colormap for the image panels, by default "afmhot".
        figsize : tuple of float, optional
            Figure size, by default (15, 5).
        pred_max : bool, optional
            If True, scale color limits to the prediction's min/max, by default True.
        output : str, optional
            Filename (or path) to save the figure to. If only a filename is given,
            it is saved under self.output.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure.
        ax : ndarray of matplotlib.axes.Axes
            The four plot axes.
        """
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
        """
        Write a denoised prediction to an XYZ file.

        Parameters
        ----------
        pred : array_like
            Denoised 2D image to write.
        apix : float
            Pixel size to use for the output coordinates.
        name : str
            Output filename. If only a filename is given, it is saved under self.output.
        """
        if len(name.split("/")) == 1:
            name = self.output + "/" + name
        img = AFMImage()
        img.write_image(pred, apix, name)
