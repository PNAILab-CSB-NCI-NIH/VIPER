import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import median_filter, uniform_filter1d, binary_closing, \
    binary_opening, binary_dilation, label, find_objects

def detect_stripe_mask_signed(img, k=4.0, k_pos=None, k_neg=None, row_window=5, smooth_len=9):
    """
    Detect signed stripe artifacts in an image via robust row-wise residuals.

    Parameters
    ----------
    img : array_like
        Input 2D image.
    k : float, optional
        Default threshold multiplier (in robust sigma) for both polarities, by default 4.0.
    k_pos : float, optional
        Threshold multiplier for positive stripes. Defaults to k.
    k_neg : float, optional
        Threshold multiplier for negative stripes. Defaults to k.
    row_window : int, optional
        Window size for the row-wise median filter, by default 5.
    smooth_len : int, optional
        Length of the 1D smoothing filter applied along rows, by default 9.
    
    Returns
    -------
    combined_mask : ndarray of bool
        Mask of all detected stripe pixels (positive or negative).
    pos_mask : ndarray of bool
        Mask of positively-deviating stripe pixels.
    neg_mask : ndarray of bool
        Mask of negatively-deviating stripe pixels.
    residual : ndarray
        Smoothed residual image used for detection.
    """
    k_pos = k if k_pos is None else k_pos
    k_neg = k if k_neg is None else k_neg

    local_med = median_filter(img, size=(row_window, 1), mode='reflect')
    residual = img - local_med
    smoothed = uniform_filter1d(residual, size=smooth_len, axis=1, mode='reflect')

    mad = np.median(np.abs(smoothed - np.median(smoothed)))
    sigma_robust = 1.5*mad

    pos_mask = smoothed >  k_pos*sigma_robust
    neg_mask = smoothed < -k_neg*sigma_robust
    combined_mask = pos_mask | neg_mask

    return combined_mask, pos_mask, neg_mask, residual

def clean_mask(mask, close_size=(1, 7), open_size=(1, 3), dilate=1):
    """
    Clean a binary mask via morphological closing, opening, and dilation.
    
    Parameters
    ----------
    mask : ndarray of bool
        Input binary mask.
    close_size : tuple of int, optional
        Structuring element size for binary closing, by default (1, 7).
    open_size : tuple of int, optional
        Structuring element size for binary opening, by default (1, 3).
    dilate : int, optional
        Number of dilation iterations, by default 1.

    Returns
    -------
    ndarray of bool
        Cleaned binary mask.
    """
    mask = binary_closing(mask, structure=np.ones(close_size))
    mask = binary_opening(mask, structure=np.ones(open_size))
    if dilate > 0:
        mask = binary_dilation(mask, iterations=dilate)
    return mask

def filter_small_components_by_extent(mask, min_length=5, min_thickness=1):
    """
    Remove connected components smaller than given row/column extents.

    Parameters
    ----------
    mask : ndarray of bool
        Input binary mask.
    min_length : int, optional
        Minimum column extent (length) required to keep a component, by default 5.
    min_thickness : int, optional
        Minimum row extent (thickness) required to keep a component, by default 1.

    Returns
    -------
    ndarray of bool
        Mask with small components removed.
    """
    labeled, n = label(mask)
    if n == 0:
        return mask

    slices = find_objects(labeled)
    filtered = np.zeros_like(mask)

    for i, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        row_extent = sl[0].stop - sl[0].start
        col_extent = sl[1].stop - sl[1].start
        if col_extent >= min_length and row_extent >= min_thickness:
            filtered |= (labeled == i)

    return filtered

def get_stripe_masks(
        img, k=4.0, k_pos=None, k_neg=None, row_window=5, smooth_len=9,
        close_size=(1, 7), open_size=(1, 3), dilate=1, min_length=5, min_thickness=1):
    """
    Detect, clean, and filter signed stripe masks from an image.

    Parameters
    ----------
    img : array_like
        Input 2D image.
    k : float, optional
        Default threshold multiplier (in robust sigma) for both polarities, by default 4.0.
    k_pos : float, optional
        Threshold multiplier for positive stripes. Defaults to k.
    k_neg : float, optional
        Threshold multiplier for negative stripes. Defaults to k.
    row_window : int, optional
        Window size for the row-wise median filter, by default 5.
    smooth_len : int, optional
        Length of the 1D smoothing filter applied along rows, by default 9.
    close_size : tuple of int, optional
        Structuring element size for binary closing, by default (1, 7).
    open_size : tuple of int, optional
        Structuring element size for binary opening, by default (1, 3).
    dilate : int, optional
        Number of dilation iterations, by default 1.
    min_length : int, optional
        Minimum column extent required to keep a component, by default 5.
    min_thickness : int, optional
        Minimum row extent required to keep a component, by default 1.
    
    Returns
    -------
    combined : ndarray of bool
        Cleaned and filtered mask of all stripe pixels.
    pos_mask : ndarray of bool
        Cleaned and filtered mask of positive stripe pixels.
    neg_mask : ndarray of bool
        Cleaned and filtered mask of negative stripe pixels.
    residual : ndarray
        Smoothed residual image used for detection.
    """
    combined, pos_mask, neg_mask, residual = detect_stripe_mask_signed(
        img, k=k, k_pos=k_pos, k_neg=k_neg, row_window=row_window, smooth_len=smooth_len)
    pos_mask = clean_mask(pos_mask, close_size, open_size, dilate)
    neg_mask = clean_mask(neg_mask, close_size, open_size, dilate)

    pos_mask = filter_small_components_by_extent(pos_mask, min_length, min_thickness)
    neg_mask = filter_small_components_by_extent(neg_mask, min_length, min_thickness)

    combined = pos_mask | neg_mask
    return combined, pos_mask, neg_mask, residual

def mask_to_boxes(mask, min_area=4, pad=1, img_shape=None):
    """
    Convert a binary mask into padded bounding boxes around connected components.

    Parameters
    ----------
    mask : ndarray of bool
        Input binary mask.
    min_area : int, optional
        Minimum component area (in pixels) required to keep a box, by default 4.
    pad : int, optional
        Number of pixels to pad each box by, by default 1.
    img_shape : tuple of int, optional
        Shape (H, W) used to clip boxes to image bounds. Defaults to mask.shape.

    Returns
    -------
    list of tuple
        List of (r0, r1, c0, c1) bounding boxes.
    """
    labeled, n = label(mask)
    boxes = []
    if n == 0:
        return boxes

    slices = find_objects(labeled)
    H, W = img_shape if img_shape is not None else mask.shape

    for i, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        component_area = np.sum(labeled[sl] == i)
        if component_area < min_area:
            continue
        r0, r1 = sl[0].start - pad, sl[0].stop + pad
        c0, c1 = sl[1].start - pad, sl[1].stop + pad
        r0, c0 = max(r0, 0), max(c0, 0)
        r1, c1 = min(r1, H), min(c1, W)
        boxes.append((r0, r1, c0, c1))

    return boxes

def plot_stripe_boxes(
        img, pos_mask, neg_mask, min_area=4, pad=1,
        cmap_img='afmhot', pos_color='red', neg_color='blue',
        linewidth=1.5, figsize=(8, 7), title=None):
    """
    Plot an image with bounding boxes overlaid around positive and negative stripe regions.

    Parameters
    ----------
    img : array_like
        Input 2D image to display.
    pos_mask : ndarray of bool
        Mask of positive stripe pixels.
    neg_mask : ndarray of bool
        Mask of negative stripe pixels.
    min_area : int, optional
        Minimum component area required to draw a box, by default 4.
    pad : int, optional
        Padding applied to each box, by default 1.
    cmap_img : str, optional
        Colormap for the image, by default 'afmhot'.
    pos_color : str, optional
        Box edge color for positive stripes, by default 'red'.
    neg_color : str, optional
        Box edge color for negative stripes, by default 'blue'.
    linewidth : float, optional
        Box edge linewidth, by default 1.5.
    figsize : tuple of float, optional
        Figure size, by default (8, 7).
    title : str, optional
        Plot title. Defaults to an auto-generated summary.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    ax : matplotlib.axes.Axes
        The plot axes.
    pos_boxes : list of tuple
        Bounding boxes for positive stripe segments.
    neg_boxes : list of tuple
        Bounding boxes for negative stripe segments.
    """
    pos_boxes = mask_to_boxes(pos_mask, min_area=min_area, pad=pad, img_shape=img.shape)
    neg_boxes = mask_to_boxes(neg_mask, min_area=min_area, pad=pad, img_shape=img.shape)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, cmap=cmap_img, origin="lower")

    for (r0, r1, c0, c1) in pos_boxes:
        rect = patches.Rectangle(
            (c0, r0), c1 - c0, r1 - r0, linewidth=linewidth,
            edgecolor=pos_color, facecolor='none')
        ax.add_patch(rect)

    for (r0, r1, c0, c1) in neg_boxes:
        rect = patches.Rectangle(
            (c0, r0), c1 - c0, r1 - r0, linewidth=linewidth,
            edgecolor=neg_color, facecolor='none')
        ax.add_patch(rect)

    if title is None:
        title = f'{len(pos_boxes)} positive ({pos_color}) / {len(neg_boxes)} negative ({neg_color}) stripe segments'
    ax.set_title(title)
    ax.axis('off')

    plt.tight_layout()
    plt.show()
    return fig, ax, pos_boxes, neg_boxes

def plot_stripe_boxes_diagnostic(
        img, k=4.0, k_pos=None, k_neg=None, row_window=5, smooth_len=9,
        close_size=(1, 7), open_size=(1, 3), dilate=1,
        min_length=5, min_thickness=1, min_area=4, box_pad=1,
        cmap_img='afmhot', pos_color='red', neg_color='blue',
        linewidth=3.5, figsize=(16, 6), plot=True, output="."):
    """
    Detect stripe masks and plot a diagnostic view of the residual and boxed image.
    
    Parameters
    ----------
    img : array_like
        Input 2D image.
    k : float, optional
        Default threshold multiplier (in robust sigma) for both polarities, by default 4.0.
    k_pos : float, optional
        Threshold multiplier for positive stripes. Defaults to k.
    k_neg : float, optional
        Threshold multiplier for negative stripes. Defaults to k.
    row_window : int, optional
        Window size for the row-wise median filter, by default 5.
    smooth_len : int, optional
        Length of the 1D smoothing filter applied along rows, by default 9.
    close_size : tuple of int, optional
        Structuring element size for binary closing, by default (1, 7).
    open_size : tuple of int, optional
        Structuring element size for binary opening, by default (1, 3).
    dilate : int, optional
        Number of dilation iterations, by default 1.
    min_length : int, optional
        Minimum column extent required to keep a component, by default 5.
    min_thickness : int, optional
        Minimum row extent required to keep a component, by default 1.
    min_area : int, optional
        Minimum component area required to draw a box, by default 4.
    box_pad : int, optional
        Padding applied to each box, by default 1.
    cmap_img : str, optional
        Colormap for the image panel, by default 'afmhot'.
    pos_color : str, optional
        Box edge color for positive stripes, by default 'red'.
    neg_color : str, optional
        Box edge color for negative stripes, by default 'blue'.
    linewidth : float, optional
        Box edge linewidth, by default 3.5.
    figsize : tuple of float, optional
        Figure size, by default (16, 6).
    plot : bool, optional
        If False, skip plotting and return masks only, by default True.
    output : str, optional
        Output directory for the saved figure, by default ".".
    
    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The created diagnostic figure, or None if plot=False.
    axes : ndarray of matplotlib.axes.Axes or None
        The plot axes, or None if plot=False.
    combined_mask : ndarray of bool
        Mask of all detected stripe pixels.
    pos_mask : ndarray of bool
        Mask of positive stripe pixels.
    neg_mask : ndarray of bool
        Mask of negative stripe pixels.
    residual : ndarray
        Smoothed residual image used for detection.
    """
    combined_mask, pos_mask, neg_mask, residual = get_stripe_masks(
        img, k=k, k_pos=k_pos, k_neg=k_neg, row_window=row_window, smooth_len=smooth_len,
        close_size=close_size, open_size=open_size, dilate=dilate,
        min_length=min_length, min_thickness=min_thickness,
    )

    if not plot:
        return None, None, combined_mask, pos_mask, neg_mask, residual

    pos_boxes = mask_to_boxes(pos_mask, min_area=min_area, pad=box_pad, img_shape=img.shape)
    neg_boxes = mask_to_boxes(neg_mask, min_area=min_area, pad=box_pad, img_shape=img.shape)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    vmax = np.percentile(np.abs(residual), 99)
    im0 = axes[0].imshow(residual, origin="lower", cmap='seismic', vmin=-vmax, vmax=vmax)
    axes[0].set_title('Residual (detector input)')
    axes[0].axis('off')
    fig.colorbar(im0, ax=axes[0], fraction=0.05, pad=0.04)

    axes[1].imshow(img, origin="lower", cmap=cmap_img)
    for (r0, r1, c0, c1) in pos_boxes:
        axes[1].add_patch(patches.Rectangle(
            (c0, r0), c1 - c0, r1 - r0, linewidth=linewidth,
            edgecolor=pos_color, facecolor='none'))
    for (r0, r1, c0, c1) in neg_boxes:
        axes[1].add_patch(patches.Rectangle(
            (c0, r0), c1 - c0, r1 - r0, linewidth=linewidth,
            edgecolor=neg_color, facecolor='none'))
    masked_pct = combined_mask.mean()*100
    axes[1].set_title(f'{len(pos_boxes)} pos / {len(neg_boxes)} neg segments  ({masked_pct:.2f}% masked)')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(f"{output}/stripe_masking.pdf", dpi=600)
    plt.show()

    return fig, axes, combined_mask, pos_mask, neg_mask, residual
