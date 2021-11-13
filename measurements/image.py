import numpy as np
from skimage.exposure import rescale_intensity as sk_rescale_intensity
from skimage.color import gray2rgb, rgb2gray

from .color import plt_color_wheel

def pad_data(data, shape=(256,256), mode='minimum'):
    """
    Padding data array

    Params
    ------
    data    : numpy_array, shape [..., width, height]
    shape   : desired shape e.g. (256, 256)
    mode    : numpy padding mode, default='minimum'

    Return
    ------
    padded_data
    """
    assert data.ndim >= 2, \
    f"Data need to have at least 2 dimensions, data.shape: {data.shape}"

    padding = [(0,0) for i in range(data.ndim)]

    width, height = data.shape[-2:]
    w_pad = max(shape[0] - width, 0) // 2
    w_pad = (w_pad, max(shape[0] - width - w_pad, 0))
    h_pad = max(shape[1] - height, 0) // 2
    h_pad = (h_pad, max(shape[1] - height - h_pad, 0))

    padding[-1] = h_pad
    padding[-2] = w_pad

    padded_data = np.pad(data, padding, mode=mode)

    return padded_data

def rescale_intensity(data, out_range=np.uint8):
    """
    Rescale Data Intensity using numpy datatype

    Params
    ------
    data        : input data, shape [..., width, height]
    out_range   : the numpy datatype used to define output_range

    Return
    ------
    r_data      : rescaled_data
    """
    if data.ndim > 2:
        r_data = np.array([rescale_intensity(x, out_range=out_range) for x in data])
    else:
        r_data = sk_rescale_intensity(data, out_range=out_range).astype(out_range)

    return r_data


def mask_data(image, mask):
    """
    Apply mask to the input image

    Params
    ------
    image   : input image, rgb [..., width, height, 3] or grayscale [..., width, height]
    mask    : input mask, [..., width, height], background pixels indexed 0
            : the labels are indexed [1, 2, 3, ...]

    Return
    ------
    image_masked
    """
    if min(image.ndim, mask.ndim) > 2:
        image_masked = np.array([mask_data(img, msk) for img, msk in zip(image, mask)])
    else:
        image = rgb2gray(image)
        image = rescale_intensity(image, out_range=np.uint8).astype(np.uint8)
        image = pad_data(image, shape=mask.shape[-2:])
        image_rgb = gray2rgb(image)
        image_masked = np.copy(image_rgb)

        labels = get_labels(mask)
        colors = plt_color_wheel().get_RGB_wheel(n_colors=len(labels))
        for i, label in enumerate(labels):
            image_masked[np.where(mask==label)] = colors[i]

    return image_masked


def get_labels(mask, background=0):
    """
    Utility for getting labels of the mask
        - background is labeled 0.
    """
    labels = np.unique(mask)
    labels = labels[labels!=background]
    return labels
