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


def gaussian_filter(
    image,
    sigma,
    truncate=4.0,
    radius=None,
    periodic_axes=None,
    is_numpy=True
):
    if is_numpy:
        image = torch.from_numpy(image)
    ndim = image.ndim

    if periodic_axes is None:
        periodic_axes = (False, ) * ndim
    elif isinstance(periodic_axes, bool):
        periodic_axes = (periodic_axes, ) * ndim
    elif len(periodic_axes) != ndim:
        raise ValueError(f"periodic_axes must be a bool or a sequence of {ndim} bools, got {len(periodic_axes)}")

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
    periodic_padding = [padding_size if periodic else 0 for periodic in periodic_axes for _ in range(2)]
    reflect_padding = [padding_size if not periodic else 0 for periodic in periodic_axes for _ in range(2)]
    image = image.unsqueeze(0).unsqueeze(0)
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    image = pad(image, periodic_padding, mode='circular')
    image = pad(image, reflect_padding, mode='reflect')

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
