import torch
from torchvision.transforms._functional_tensor import _get_gaussian_kernel1d
from torch.nn.functional import pad, conv1d, conv2d, conv3d


def get_gaussian_kernel1d(size, sigma, dtype=torch.float32, device="cpu"):
    return _get_gaussian_kernel1d(size, sigma, dtype=dtype, device=device)


def get_gaussian_kernel2d(size, sigma, dtype=torch.float32, device="cpu"):
    kernel_x = get_gaussian_kernel1d(size, sigma, dtype=dtype, device=device)
    kernel_y = get_gaussian_kernel1d(size, sigma, dtype=dtype, device=device)
    return kernel_x[:, None] * kernel_y[None, :]


def get_gaussian_kernel3d(size, sigma, dtype=torch.float32, device="cpu"):
    kernel_x = get_gaussian_kernel1d(size, sigma, dtype=dtype, device=device)
    kernel_y = get_gaussian_kernel1d(size, sigma, dtype=dtype, device=device)
    kernel_z = get_gaussian_kernel1d(size, sigma, dtype=dtype, device=device)
    return kernel_x[:, None, None] * kernel_y[None, :, None] * kernel_z[None, None, :]


def gaussian_filter(image, sigma, truncate=4.0, radius=None, mode='reflect', is_numpy=False):
    if is_numpy:
        image = torch.from_numpy(image)
    ndim = image.ndim
    if radius is None:
        radius = round(truncate * sigma)
    kernel_size = 2*radius + 1
    if ndim == 1:
        kernel = get_gaussian_kernel1d(kernel_size, sigma, dtype=image.dtype, device=image.device)
    elif ndim == 2:
        kernel = get_gaussian_kernel2d(kernel_size, sigma, dtype=image.dtype, device=image.device)
    elif ndim == 3:
        kernel = get_gaussian_kernel3d(kernel_size, sigma, dtype=image.dtype, device=image.device)
    else:
        raise ValueError(f"Gaussian filter only supports 1D, 2D, or 3D arrays, got {ndim}D")
    padding_size = radius
    padding = (padding_size, ) * 2 * ndim
    image = image.unsqueeze(0).unsqueeze(0)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    image = pad(image, padding, mode=mode)
    if ndim == 1:
        image = conv1d(image, kernel, padding=0)
    elif ndim == 2:
        image = conv2d(image, kernel, padding=0)
    elif ndim == 3:
        image = conv3d(image, kernel, padding=0)
    image = image.squeeze(0).squeeze(0)
    if is_numpy:
        image = image.numpy()
    return image
