
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
        self.img = None
        self.apix = apix
        self.n_pixels = None

        if path is not None:
            self.read_path(path, unit)
        if tensor is not None:
            self.read_tensor(tensor)

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else: self.device = device
    
    def read_xyz_file_old(self, path):
        with open(path, "r") as f:
            first_line = f.readline().strip().split()

        is_numeric = True
        for token in first_line:
            try: float(token)
            except ValueError:
                is_numeric = False
                break

        if is_numeric:
            df = pd.read_csv(path, sep=r"\s+", header=None, names=["x", "y", "z"])
        else: df = pd.read_csv(path, sep=r"\s+")
        return df

    def read_path_old(self, path, unit="A"):
        # Read file
        df = self.read_xyz_file(path)

        # Get unique x and y values to determine grid dimensions
        x_unique = df['x'].unique()
        y_unique = df['y'].unique()

        # Calculate pixel sizes
        x_pixel_size = (x_unique[-1] - x_unique[0])/(len(x_unique) - 1)
        y_pixel_size = (y_unique[-1] - y_unique[0])/(len(y_unique) - 1)

        if self.apix is not None:
            x_pixel_size = self.apix
            y_pixel_size = self.apix
        elif x_pixel_size != y_pixel_size:
            if math.isclose(x_pixel_size, y_pixel_size, rel_tol=1e-4):
                x_pixel_size = 0.5*(x_pixel_size + y_pixel_size)
                y_pixel_size = x_pixel_size
            else:
                raise(f"X and Y spacing are not the same! {x_pixel_size} vs {y_pixel_size}")
        
        # Reshape z values into a 2D grid
        Z = df['z'].values.reshape(len(y_unique), len(x_unique))
        self.read_tensor(Z, x_pixel_size, unit)

    def read_xyz_file(self, path):
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
        norm = 1.
        if unit == "A": norm = 1.
        if unit == "nm": norm = 10.
        self.img = torch.as_tensor(tensor).float()*norm
        self.apix = apix*norm
        self.n_pixels = self.img.shape[0]

    def normalize_std(self, img):
        return (img - img.mean())/img.std()

    def normalize_1(self, img):
        return (img - img.min())/(img.max() - img.min())
    
    def write_image(self, tensor=None, apix=None, name='output.txt'):
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
        if img is None: img = self.img
        iplot = img.detach().cpu().numpy()
        plt.imshow(iplot)
        plt.colorbar(label='Height (Å)')
        plt.show()

    def ncc(self, img, target):
        img_norm = img - img.mean()
        tar_norm = target - target.mean()
        num = torch.sum(img_norm*tar_norm)
        den = torch.sqrt(torch.sum(img_norm**2)*torch.sum(tar_norm**2) + 1e-8)
        return num/den
    
    def cosine(self, img, target):
        img_flat = img.flatten()
        tar_flat = target.flatten()
        sim = torch.nn.functional.cosine_similarity(img_flat, tar_flat, dim=0)
        return sim
    
    def cc_fft(self, img, target):
        img_mean = img - img.mean()
        tar_mean = target - target.mean()
        Fimg = fft.fftn(img_mean.float())
        Ftar = fft.fftn(tar_mean.float())
        CC_freq = Fimg*torch.conj(Ftar)
        CC = fft.ifftn(CC_freq).real
        return CC.mean()
    
    def mse(self, img, target):
        return F.mse_loss(img, target)
