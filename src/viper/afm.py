
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.fft as fft
import torch.nn.functional as F
import math
from pathlib import Path

class AFMImage():
    def __init__(self, path=None, unit=None, tensor=None, apix=None, device=None):
        """
        Initialize an AFMImage object from a file path or tensor.

        Parameters
        ----------
        path : str or Path, optional
            Path to an XYZ-format file to load.
        unit : str, optional
            Unit of the input coordinates ("A" or "nm").
        tensor : array_like, optional
            Pre-loaded 2D height map to use directly.
        apix : float, optional
            Pixel size (Angstroms per pixel), overrides computed spacing.
        device : str, optional
            Device to use ('cuda' or 'cpu'). Defaults to CUDA if available.
        """
        self.img = None
        self.apix = apix
        self.n_pixels = None

        if path is not None:
            self.read_path(path, unit)
        if tensor is not None:
            self.read_tensor(tensor, apix, unit)

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else: self.device = device

    def read_xyz_file(self, path):
        """
        Read and validate a whitespace-delimited XYZ file into a DataFrame.

        Parameters
        ----------
        path : str or Path
            Path to the XYZ file.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns "x", "y", "z", containing finite values.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found:\n{path}")

        # Support numeric files and files with a single header line.
        try:
            data = np.loadtxt(path, ndmin=2)
        except ValueError:
            data = np.loadtxt(path, skiprows=1, ndmin=2)

        if data.shape[1] < 3:
            raise ValueError(
                f"{path.name}: expected at least 3 columns (X, Y, Z), "
                f"but found shape {data.shape}."
            )

        xyz = np.asarray(data[:, :3], dtype=float)
        if not np.isfinite(xyz).all():
            raise ValueError(f"{path.name}: X, Y, or Z contains NaN/Inf.")

        # Preserve the DataFrame return type.
        return pd.DataFrame(xyz, columns=["x", "y", "z"])

    def read_path(self, path, unit="A"):
        """
        Load an XYZ file, reconstruct a uniformly-spaced 2D grid, and store it as the image tensor.

        Parameters
        ----------
        path : str or Path
            Path to the XYZ file.
        unit : str, optional
            Unit of the input coordinates ("A" or "nm"), by default "A".
        """
        df = self.read_xyz_file(path)

        x = df["x"].to_numpy()
        y = df["y"].to_numpy()
        z = df["z"].to_numpy()

        # np.unique sorts coordinates in ascending order.
        x_unique = np.unique(x)
        y_unique = np.unique(y)
        nx, ny = len(x_unique), len(y_unique)

        if nx < 2 or ny < 2:
            raise ValueError("Need at least two coordinates along each axis.")

        if nx * ny != len(z):
            raise ValueError(
                f"Incomplete grid: unique X={nx}, Y={ny}, "
                f"expected {nx * ny} points, found {len(z)}."
            )

        # Place each height at its actual coordinate.
        ix = np.searchsorted(x_unique, x)
        iy = np.searchsorted(y_unique, y)

        Z = np.full((ny, nx), np.nan, dtype=float)
        Z[iy, ix] = z

        if np.isnan(Z).any():
            raise ValueError("Grid contains missing points or duplicate coordinates.")

        # Validate uniform spacing before representing it with one pixel size.
        dx = np.diff(x_unique)
        dy = np.diff(y_unique)

        if not np.allclose(dx, dx.mean(), rtol=1e-4, atol=0):
            raise ValueError("X coordinates are not uniformly spaced.")
        if not np.allclose(dy, dy.mean(), rtol=1e-4, atol=0):
            raise ValueError("Y coordinates are not uniformly spaced.")

        x_pixel_size = float(dx.mean())
        y_pixel_size = float(dy.mean())

        # Preserve your explicit pixel-size override.
        if self.apix is not None:
            x_pixel_size = self.apix
            y_pixel_size = self.apix
        elif np.isclose(
            x_pixel_size, y_pixel_size, rtol=1e-4, atol=0
        ):
            pixel_size = 0.5 * (x_pixel_size + y_pixel_size)
            x_pixel_size = y_pixel_size = pixel_size
        else:
            raise ValueError(
                "X and Y spacing are not the same! "
                f"{x_pixel_size} vs {y_pixel_size}"
            )

        self.read_tensor(Z, x_pixel_size, unit)
    
    def read_tensor(self, tensor, apix, unit="A"):
        """
        Store a 2D tensor as the image, converting units and setting pixel size.

        Parameters
        ----------
        tensor : array_like
            2D height map to store.
        apix : float
            Pixel size in the given unit.
        unit : str, optional
            Unit of apix ("A" or "nm"), by default "A".
        """
        norm = 1.
        if unit == "A": norm = 1.
        if unit == "nm": norm = 10.
        self.img = torch.as_tensor(tensor).float()*norm
        self.apix = apix*norm
        self.n_pixels = self.img.shape[0]

    def normalize_std(self, img):
        """
        Normalize an image to zero mean and unit standard deviation.

        Parameters
        ----------
        img : torch.Tensor
            Input image.

        Returns
        -------
        torch.Tensor
            Standardized image.
        """
        return (img - img.mean())/img.std()

    def normalize_1(self, img):
        """
        Normalize an image to the range [0, 1].

        Parameters
        ----------
        img : torch.Tensor
            Input image.

        Returns
        -------
        torch.Tensor
            Min-max normalized image.
        """
        return (img - img.min())/(img.max() - img.min())
    
    def write_image(self, tensor=None, apix=None, name='output.txt'):
        """
        Write a 2D height map to a tab-delimited XYZ text file.

        Parameters
        ----------
        tensor : array_like, optional
            2D height map to write. Defaults to self.img.
        apix : float, optional
            Pixel size used to generate x/y coordinates. Defaults to self.apix.
        name : str, optional
            Output file path, by default 'output.txt'.
        """
        if tensor is None: tensor = self.img
        if apix is None: apix = self.apix

        # Convert to numpy if it's a tensor
        if torch.is_tensor(tensor):
            Z = tensor.cpu().numpy()
        else: Z = np.asarray(tensor)
        
        height, width = Z.shape
        
        # Create coordinate grids
        x_coords = np.arange(width)*apix
        y_coords = np.arange(height)*apix
        
        # Open file and write header
        with open(name, 'w') as f:
            f.write('x\ty\tz\n')
            
            # Write data row by row
            for i, y in enumerate(y_coords):
                for j, x in enumerate(x_coords):
                    z = Z[i, j]
                    f.write(f'{x:.5f}\t{y:.5f}\t{z:.6f}\n')

    def view_image(self, img=None):
        """
        Display a 2D image using matplotlib with a height colorbar.

        Parameters
        ----------
        img : torch.Tensor, optional
            Image to display. Defaults to self.img.
        """
        if img is None: img = self.img
        iplot = img.detach().cpu().numpy()
        plt.imshow(iplot)
        plt.colorbar(label='Height (Å)')
        plt.show()

    def ncc(self, img, target):
        """
        Compute the normalized cross-correlation between two images.

        Parameters
        ----------
        img : torch.Tensor
            First input image.
        target : torch.Tensor
            Second input image to compare against.

        Returns
        -------
        torch.Tensor
            Scalar normalized cross-correlation value.
        """
        img_norm = img - img.mean()
        tar_norm = target - target.mean()
        num = torch.sum(img_norm*tar_norm)
        den = torch.sqrt(torch.sum(img_norm**2)*torch.sum(tar_norm**2) + 1e-8)
        return num/den
    
    def cosine(self, img, target):
        """
        Compute the cosine similarity between two flattened images.
        
        Parameters
        ----------
        img : torch.Tensor
            First input image.
        target : torch.Tensor
            Second input image to compare against.

        Returns
        -------
        torch.Tensor
            Scalar cosine similarity value.
        """
        img_flat = img.flatten()
        tar_flat = target.flatten()
        sim = torch.nn.functional.cosine_similarity(img_flat, tar_flat, dim=0)
        return sim
    
    def cc_fft(self, img, target):
        """
        Compute the mean cross-correlation between two images via FFT.

        Parameters
        ----------
        img : torch.Tensor
            First input image.
        target : torch.Tensor
            Second input image to compare against.

        Returns
        -------
        torch.Tensor
            Mean real-valued cross-correlation computed in frequency space.
        """
        img_mean = img - img.mean()
        tar_mean = target - target.mean()
        Fimg = fft.fftn(img_mean.float())
        Ftar = fft.fftn(tar_mean.float())
        CC_freq = Fimg*torch.conj(Ftar)
        CC = fft.ifftn(CC_freq).real
        return CC.mean()
    
    def mse(self, img, target):
        """
        Compute the mean squared error between two images.
        
        Parameters
        ----------
        img : torch.Tensor
            First input image.
        target : torch.Tensor
            Second input image to compare against.

        Returns
        -------
        torch.Tensor
            Scalar mean squared error.
        """
        return F.mse_loss(img, target)
