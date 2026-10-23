import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

from afm import AFMImage
from stripes import plot_stripe_boxes_diagnostic

class Dataset:
    def __init__(self, path=None, unit="A"):
        """
        Initialize a Dataset, optionally loading trace/retrace AFM images from a path.

        Parameters
        ----------
        path : str, optional
            Directory containing HeightTrace/HeightRetrace image files.
        unit : str, optional
            Unit of the input coordinates ("A" or "nm"), by default "A".
        """
        if path is not None:
            self.read_path(path, unit)

    def read_files(self, path, files, is_trace, imgs):
        """
        Load a list of AFM image files into a nested image dictionary.

        Parameters
        ----------
        path : str
            Directory containing the files.
        files : list of str
            Filenames to load.
        is_trace : bool
            If True, treat files as "HeightTrace" images; otherwise "HeightRetrace".
        imgs : dict
            Nested dictionary to populate, with "trace"/"retrace" keys.
        
        Returns
        -------
        dict
            Updated imgs dictionary with loaded image data.
        """
        rep = "HeightTrace" if is_trace else "HeightRetrace"
        key = "trace" if is_trace else "retrace"
        for f in files:
            path_i = f"{path}/{f}"
            afm = AFMImage(path_i)
            img_i = afm.img
            apix_i = afm.apix
            imgs[key][f.replace(rep, "")] = {
                "path": path_i, "img": img_i.clone().detach(),
                "apix": apix_i, "name": f
            }
        return imgs

    def fuse_files(self, imgs):
        """
        Pair matching trace and retrace images by shared filename key.

        Parameters
        ----------
        imgs : dict
            Nested dictionary with "trace" and "retrace" image entries.
        
        Returns
        -------
        X : list of torch.Tensor
            Trace images.
        y : list of torch.Tensor
            Retrace images.
        apix : list of float
            Pixel sizes for each paired image.
        name : list of str
            Original filenames.
        mx : list of torch.Tensor
            Empty boolean masks for trace images.
        my : list of torch.Tensor
            Empty boolean masks for retrace images.
        """
        X, y, apix, name = [], [], [], []
        mx, my = [], []
        for tkey in imgs["trace"].keys():
            if tkey in imgs["retrace"].keys():
                X.append(imgs["trace"][tkey]["img"])
                y.append(imgs["retrace"][tkey]["img"])
                apix.append(imgs["trace"][tkey]["apix"])
                name.append(imgs["trace"][tkey]["name"])
                mx.append(torch.zeros_like(imgs["trace"][tkey]["img"]).bool())
                my.append(torch.zeros_like(imgs["trace"][tkey]["img"]).bool())
        return X, y, apix, name, mx, my

    def read_path(self, path, unit="A"):
        """
        Load all trace/retrace AFM images from a directory and stack them into tensors.

        Parameters
        ----------
        path : str
            Directory containing HeightTrace/HeightRetrace image files.
        unit : str, optional
            Unit of the input coordinates ("A" or "nm"), by default "A".
        """
        files_x = [f for f in os.listdir(path) if "HeightTrace" in f]
        files_y = [f for f in os.listdir(path) if "HeightRetrace" in f]

        imgs = {"trace": {}, "retrace": {}}
        imgs = self.read_files(path, files_x, True, imgs)
        imgs = self.read_files(path, files_y, False, imgs)
        self.imgs_x, self.imgs_y, self.apix, self.names, \
            self.mask_x, self.mask_y = self.fuse_files(imgs)
        
        self.X_raw = torch.stack(self.imgs_x)
        self.y_raw = torch.stack(self.imgs_y)
        self.n_images, self.height, self.width = self.X_raw.shape
        print(f"Loaded Dataset (n, h, w): {self.X_raw.shape}")

    def reset_stripe_masks(self):
        """Reset all trace and retrace stripe masks to all-False."""
        for i in range(len(self.imgs_y)):
            self.mask_x[i] = torch.zeros_like(self.imgs_x[i]).bool()
            self.mask_y[i] = torch.zeros_like(self.imgs_y[i]).bool()

    def make_stripe_masks(
            self, idx=None, k=3.5, k_pos=None, k_neg=None, row_window=4,
            smooth_len=9, min_length=10, min_thickness=1, min_area=10, output="."):
        """
        Detect and store stripe masks for all trace/retrace image pairs.

        Parameters
        ----------
        idx : int, optional
            Index of the image for which to display the diagnostic plot.
        k : float, optional
            Default threshold multiplier (in robust sigma) for both polarities, by default 3.5.
        k_pos : float, optional
            Threshold multiplier for positive stripes. Defaults to k.
        k_neg : float, optional
            Threshold multiplier for negative stripes. Defaults to k.
        row_window : int, optional
            Window size for the row-wise median filter, by default 4.
        smooth_len : int, optional
            Length of the 1D smoothing filter applied along rows, by default 9.
        min_length : int, optional
            Minimum column extent required to keep a stripe component, by default 10.
        min_thickness : int, optional
            Minimum row extent required to keep a stripe component, by default 1.
        min_area : int, optional
            Minimum component area required to draw a box, by default 10.
        output : str, optional
            Output directory for saved diagnostic figures, by default ".".
        """
        for i in range(len(self.imgs_y)):
            plot = False
            fig, axes, mx, px, nx, resx = plot_stripe_boxes_diagnostic(
                self.imgs_x[i], k=k, k_pos=k_pos, k_neg=k_neg, row_window=row_window,
                smooth_len=smooth_len, min_length=min_length, min_thickness=min_thickness,
                min_area=min_area, plot=plot, output=output
            )
            if idx is not None and idx == i: plot = True
            fig, axes, my, py, ny, resy = plot_stripe_boxes_diagnostic(
                self.imgs_y[i], k=k, k_pos=k_pos, k_neg=k_neg, row_window=row_window,
                smooth_len=smooth_len, min_length=min_length, min_thickness=min_thickness,
                min_area=min_area, plot=plot, output=output
            )
            self.mask_x[i] = torch.as_tensor(mx).bool()
            self.mask_y[i] = torch.as_tensor(my).bool()

    def view_image(self, idx):
        """
        Display the trace and retrace images for a given index side by side.

        Parameters
        ----------
        idx : int
            Index of the image pair to display.
        """
        xplot = self.X_raw[idx].detach().cpu().numpy()
        yplot = self.y_raw[idx].detach().cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
        for ax, data, title in zip(axes, [xplot, yplot], ["Trace", "Retrace"]):
            im = ax.imshow(data, cmap="afmhot", origin="lower")
            ax.set_title(title)
            fig.colorbar(im, ax=ax, label="Height")
        plt.show()

    def find_cc(self, im1, im2):
        """
        Compute the normalized cross-correlation between two images.

        Parameters
        ----------
        im1 : torch.Tensor
            First input image.
        im2 : torch.Tensor
            Second input image.
        
        Returns
        -------
        torch.Tensor or float
            Normalized cross-correlation value, or 0.0 if either image has zero energy.
        """
        num = torch.sum(im1*im2)
        denom = torch.sqrt(torch.sum(im1*im1) * torch.sum(im2*im2))
        if denom == 0: return 0.0
        return num/denom

    def _best_shift_ncc(self, region, crop_x, crop_height, crop_width, low_range, high_range):
        """
        Find the shift maximizing normalized cross-correlation between a crop and a search region.

        Parameters
        ----------
        region : array_like
            Search region to scan for the best-matching crop.
        crop_x : array_like
            Reference crop to match against.
        crop_height : int
            Height of the crop window.
        crop_width : int
            Width of the crop window.
        low_range : int
            Lower bound of the shift search range.
        high_range : int
            Upper bound of the shift search range (exclusive).
        
        Returns
        -------
        best_cc : torch.Tensor
            Best normalized cross-correlation value found.
        best_dy : int
            Row shift corresponding to the best match.
        best_dx : int
            Column shift corresponding to the best match.
        best_crop_y : torch.Tensor
            Best-matching crop extracted from the search region.
        """
        # Normalize to tensors in case data is a mix of tensors and np arrays
        if not torch.is_tensor(region):
            region = torch.as_tensor(region)
        if not torch.is_tensor(crop_x):
            crop_x = torch.as_tensor(crop_x)

        # unfold(dim, size, step) applied twice builds all sliding windows
        # Result shape: (span_y, span_x, crop_height, crop_width)
        windows = region.unfold(0, crop_height, 1).unfold(1, crop_width, 1)

        # Numerator: sum(crop_x*window) for every shift -> (span_y, span_x)
        num = torch.einsum('ijkl,kl->ij', windows, crop_x)

        # Denominator: norm(crop_x)*norm(window) for every shift
        win_energy = torch.einsum('ijkl,ijkl->ij', windows, windows)
        x_energy = torch.sum(crop_x*crop_x)
        denom = torch.sqrt(x_energy*win_energy)

        cc = torch.where(denom > 0, num/denom, torch.zeros_like(denom))

        flat_idx = torch.argmax(cc)
        best_k, best_e = np.unravel_index(flat_idx.item(), cc.shape)
        best_cc = cc[best_k, best_e]
        best_dy = low_range + best_k
        best_dx = low_range + best_e
        best_crop_y = windows[best_k, best_e]

        return best_cc, best_dy, best_dx, best_crop_y

    def random_crops(
            self, crop_num=3000, crop_width=32, crop_height=32,
            low_range=-10, high_range=11, drift_mode='per_crop'):
        """
        Extract drift-corrected random crop pairs from all trace/retrace images.

        Parameters
        ----------
        crop_num : int, optional
            Number of random crops per image, by default 3000.
        crop_width : int, optional
            Width of each crop, by default 32.
        crop_height : int, optional
            Height of each crop, by default 32.
        low_range : int, optional
            Lower bound of the drift search range, by default -10.
        high_range : int, optional
            Upper bound of the drift search range (exclusive), by default 11.
        drift_mode : str, optional
            Drift estimation strategy, either 'per_crop' or 'per_image', by default 'per_crop'.
        """
        assert drift_mode in ('per_crop', 'per_image'), \
            f"drift_mode must be 'per_crop' or 'per_image', got {drift_mode!r}"

        self.crop_num = crop_num
        self.low_range = low_range
        self.high_range = high_range
        self.drift_mode = drift_mode

        margin_lo = -low_range
        margin_hi = high_range
        span = high_range - low_range

        # Valid range for crop top-left corner so that crop_x AND every
        # shifted crop_y in the search window stay inside the image
        x_min, x_max = margin_lo, self.width - crop_width - margin_hi + 1
        y_min, y_max = margin_lo, self.height - crop_height - margin_hi + 1

        self.xs = np.random.randint(x_min, x_max, size=self.crop_num, dtype=int)
        self.ys = np.random.randint(y_min, y_max, size=self.crop_num, dtype=int)

        # Crops from the "retrace" (y) images at the raw, unshifted locations —
        # kept only as a reference/baseline (no correlation-based alignment).
        crops_y_ncc = []
        for img in self.imgs_y:
            for j in range(self.crop_num):
                y0, x0 = self.ys[j], self.xs[j]
                crops_y_ncc.append(img[y0:y0 + crop_height, x0:x0 + crop_width])

        # Per_image mode: estimate one (dy, dx) per image up front, using a
        # single large-region NCC search rather than aggregating per-crop results
        self.image_drift_x = None
        self.image_drift_y = None
        if drift_mode == 'per_image':
            image_drift_x = []
            image_drift_y = []
            for i in range(self.n_images):
                img_x = self.imgs_x[i]
                img_y = self.imgs_y[i]

                # Large central region so we get a stab;e estimate
                region_h = self.height - 2*margin_lo - (span - 1)
                region_w = self.width - 2*margin_lo - (span - 1)
                region_h = max(region_h, crop_height)
                region_w = max(region_w, crop_width)
                cy0 = (self.height - region_h) // 2
                cx0 = (self.width - region_w) // 2

                ref_crop = img_x[cy0:cy0 + region_h, cx0:cx0 + region_w]
                search_region = img_y[cy0 + low_range: cy0 + low_range + region_h + span - 1,
                                      cx0 + low_range: cx0 + low_range + region_w + span - 1]

                _, best_dy, best_dx, _ = self._best_shift_ncc(
                    search_region, ref_crop, region_h, region_w, low_range, high_range)
                image_drift_x.append(best_dx)
                image_drift_y.append(best_dy)

            self.image_drift_x = torch.tensor(image_drift_x)
            self.image_drift_y = torch.tensor(image_drift_y)
            print('Per-image drift estimates -> dx:', self.image_drift_x.tolist())
            print('Per-image drift estimates -> dy:', self.image_drift_y.tolist())

        crops_x, crops_y_cc = [], []
        crops_mx, crops_my  = [], []
        drift_x, drift_y    = [], []

        for i in range(self.n_images):
            img_x = self.imgs_x[i]
            img_y = self.imgs_y[i]
            mask_x = self.mask_x[i]
            mask_y = self.mask_y[i]

            for j in range(self.crop_num):
                y0, x0 = self.ys[j], self.xs[j]

                crop_x = img_x[y0:y0 + crop_height, x0:x0 + crop_width]
                crops_x.append(crop_x)

                # Mask for x follows the exact same unshifted crop location.
                crop_mx = mask_x[y0:y0 + crop_height, x0:x0 + crop_width]
                crops_mx.append(crop_mx)

                if drift_mode == 'per_crop':
                    # Region of img_y covering every possible shifted crop at once
                    search_region = img_y[y0 + low_range: y0 + low_range + crop_height + span - 1,
                                          x0 + low_range: x0 + low_range + crop_width  + span - 1]
                    best_cc, best_dy, best_dx, best_crop_y = self._best_shift_ncc(
                        search_region, crop_x, crop_height, crop_width, low_range, high_range)
                else: # per_image: reuse this image's single drift estimate
                    best_dy = int(self.image_drift_y[i].item())
                    best_dx = int(self.image_drift_x[i].item())
                    ym0, xm0 = y0 + best_dy, x0 + best_dx
                    best_crop_y = img_y[ym0:ym0 + crop_height, xm0:xm0 + crop_width]

                crops_y_cc.append(best_crop_y)
                drift_x.append(best_dx)
                drift_y.append(best_dy)

                # Mask for y must follow the SAME drift-corrected location that best_crop_y used
                ym0, xm0 = y0 + best_dy, x0 + best_dx
                crop_my = mask_y[ym0:ym0 + crop_height, xm0:xm0 + crop_width]
                crops_my.append(crop_my)

        self.crops_x = torch.stack(crops_x)
        self.crops_y_ncc = torch.stack(crops_y_ncc)
        self.crops_y_cc = torch.stack(crops_y_cc)
        self.crops_mx = torch.stack(crops_mx)
        self.crops_my = torch.stack(crops_my)
        self.drift_x = torch.tensor(drift_x)
        self.drift_y = torch.tensor(drift_y)

        print("crops shape:", self.crops_x.shape)
        print("crops_mx shape:", self.crops_mx.shape)
        print("crops_my shape:", self.crops_my.shape)

        # Store crop dimensions separately from full-image width/height
        if self.crops_x.ndim == 4:
            _, self.crop_h, self.crop_w, self.channels = self.crops_x.shape
        else:
            _, self.crop_h, self.crop_w = self.crops_x.shape
            self.channels = 1

        self.plot_drift(show_current_cut=False)

    def plot_drift(self, bins=None, show_current_cut=True):
        """
        Plot a 2D histogram of estimated trace/retrace drift values.

        Parameters
        ----------
        bins : int or sequence, optional
            Histogram bin specification. Defaults to a range spanning the drift values.
        show_current_cut : bool, optional
            If True, overlay the current quality_cut bounds, by default True.
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure.
        ax : matplotlib.axes.Axes
            The plot axes.
        """
        dx = self.drift_x
        dy = self.drift_y
        if torch.is_tensor(dx):
            dx = dx.cpu().numpy()
        if torch.is_tensor(dy):
            dy = dy.cpu().numpy()

        if bins is None:
            span_x = int(dx.max() - dx.min()) + 1
            span_y = int(dy.max() - dy.min()) + 1
            bins = [max(span_x, 1), max(span_y, 1)]

        fig, ax = plt.subplots(figsize=(5, 4))

        h = ax.hist2d(dx, dy, bins=bins, cmap='viridis')
        fig.colorbar(h[3], ax=ax, label='crop count')
        ax.scatter(dx, dy, s=4, color='white', alpha=0.25, edgecolors='none')
        b_attrs = all(hasattr(self, a) for a in ('x_min', 'x_max', 'y_min', 'y_max'))
        if show_current_cut and b_attrs:
            x0, x1 = self.x_min, self.x_max
            y0, y1 = self.y_min, self.y_max
            ax.add_patch(plt.Rectangle(
                (x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor='red',
                linewidth=2, linestyle='--', label='current cut() bounds'))
            ax.legend(loc='upper right')

        ax.set_xlabel('drift_x')
        ax.set_ylabel('drift_y')
        ax.set_title(f'Trace/retrace drift distribution (n={len(dx)})')
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5)

        plt.tight_layout()
        plt.show()

        return fig, ax

    def fit_affine_intensity(self, crop_x, crop_y, valid_mask=None, eps=1e-8):
        """
        Fit an affine intensity transform mapping crop_y onto crop_x via least squares.

        Parameters
        ----------
        crop_x : torch.Tensor
            Reference crop.
        crop_y : torch.Tensor
            Crop to be intensity-matched to crop_x.
        valid_mask : torch.Tensor, optional
            Boolean mask selecting valid pixels to fit on. Defaults to all pixels.
        eps : float, optional
            Unused numerical safeguard, by default 1e-8.
        
        Returns
        -------
        crop_y_matched : torch.Tensor
            Intensity-matched version of crop_y.
        a : torch.Tensor
            Fitted scale coefficient.
        b : torch.Tensor
            Fitted offset coefficient.
        """
        if valid_mask is not None:
            vmask = valid_mask.bool()
            vx = crop_x[vmask].flatten()
            vy = crop_y[vmask].flatten()
        else:
            vx, vy = crop_x.flatten(), crop_y.flatten()

        min_valid_frac = 0.3
        min_valid_pixels = int(min_valid_frac * crop_x.numel())

        if vx.numel() < 2:
            return crop_y, torch.tensor(1.0), torch.tensor(0.0)
        if vx.numel() < max(min_valid_pixels, 10):
            print("skipped")
            return crop_y, torch.tensor(1.0), torch.tensor(0.0)

        A = torch.stack([vy, torch.ones_like(vy)], dim=1)
        solution = torch.linalg.lstsq(A, vx.unsqueeze(1)).solution
        a, b = solution[0, 0], solution[1, 0]

        crop_y_matched = a * crop_y + b
        return crop_y_matched, a, b

    def normalize(self, use_affine=False):
        """
        Normalize cut crops to a common intensity scale, with optional affine matching.

        Parameters
        ----------
        use_affine : bool, optional
            If True, affine-match each y crop to its x crop before normalizing, by default False.
        """
        x = self.x_cuts
        y = self.y_cuts
        has_masks = hasattr(self, 'crops_mx') and hasattr(self, 'crops_my')

        if use_affine:
            n_crops = x.shape[0]
            y_matched = torch.empty_like(y)
            affine_a = torch.empty(n_crops)
            affine_b = torch.empty(n_crops)

            for i in range(n_crops):
                if has_masks:
                    valid_mask = (1.0 - self.crops_mx[i].float()) * (1.0 - self.crops_my[i].float())
                else:
                    valid_mask = None
                y_i_matched, a, b = self.fit_affine_intensity(x[i], y[i], valid_mask=valid_mask)
                y_matched[i] = y_i_matched
                affine_a[i] = a
                affine_b[i] = b

            self.affine_a = affine_a
            self.affine_b = affine_b
            print(f'Affine intensity match:')
            print(f'-> a range [{affine_a.min():.4f}, {affine_a.max():.4f}]')
            print(f'-> b range [{affine_b.min():.4f}, {affine_b.max():.4f}]')
        else:
            y_matched = y
            self.affine_a = None
            self.affine_b = None

        x_min = x.amin()
        x_max = x.amax()
        scale = x_max - x_min

        if scale == 0:
            scale = torch.tensor(1.0)

        self.x_norm = (x - x_min)/scale
        self.y_norm = (y_matched - x_min)/scale
        self.norm_x_min = x_min
        self.norm_scale = scale

        print(f'Normalization parameters:')
        print(f'-> x_norm range: {self.x_norm.min().item()}, {self.x_norm.max().item()}')
        print(f'-> y_norm range: {self.y_norm.min().item()}, {self.y_norm.max().item()}')

    def quality_cut(self, x_min=-4, x_max=0, y_min=-2, y_max=2):
        """
        Filter crops by drift magnitude and store the retained subset.
        
        Parameters
        ----------
        x_min : int, optional
            Minimum allowed X drift, by default -4.
        x_max : int, optional
            Maximum allowed X drift, by default 0.
        y_min : int, optional
            Minimum allowed Y drift, by default -2.
        y_max : int, optional
            Maximum allowed Y drift, by default 2.
        """
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        x_mask = (self.drift_x >= x_min) & (self.drift_x <= x_max)
        y_mask = (self.drift_y >= y_min) & (self.drift_y <= y_max)
        mask = x_mask & y_mask

        ind = torch.nonzero(mask, as_tuple=True)[0]

        self.x_cuts  = self.crops_x[ind]
        self.y_cuts  = self.crops_y_cc[ind]
        self.mx_cuts = self.crops_mx[ind]
        self.my_cuts = self.crops_my[ind]
        self.drift_x = self.drift_x[ind]
        self.drift_y = self.drift_y[ind]

        print("crops shape:", self.x_cuts.shape)
        self.cuts_num = self.x_cuts.shape[0]
