import numpy as np
import torch


def create_random_field(
    shape,
    porosity=0.5,
    correlation_length=5,
    smoothing_sigma=2.0,
    periodic_axes=None,
    seed=None
):
    """
    Create a random porous field using noise + smoothing with periodic boundaries.

    Parameters:
    -----------
    shape : tuple
        Shape of the output array (e.g., (64, 64, 64) for 3D or (256, 256) for 2D)
    porosity : float, default=0.5
        Target porosity (fraction of void space), between 0 and 1
    correlation_length : float, default=5
        Controls the size of porous features (larger = bigger pores)
    smoothing_sigma : float, default=2.0
        Gaussian smoothing parameter (larger = smoother transitions)
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    np.ndarray
        Binary array where True/1 = void space, False/0 = solid
    """

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Generate correlated noise using multiple scales
    noise = np.random.randn(*shape)

    # Convert to torch tensor and add channel dimension
    noise_tensor = torch.from_numpy(noise).float()
    is_3d = len(shape) == 3

    def create_gaussian_kernel(sigma, dims):
        """Create a Gaussian kernel for convolution."""
        # Kernel size should be odd and large enough to capture the Gaussian
        kernel_size = int(2 * np.ceil(3 * sigma) + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Create coordinate grids
        if dims == 2:
            coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
            x, y = torch.meshgrid(coords, coords, indexing='ij')
            kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        else:  # 3D
            coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
            x, y, z = torch.meshgrid(coords, coords, coords, indexing='ij')
            kernel = torch.exp(-(x**2 + y**2 + z**2) / (2 * sigma**2))

        # Normalize
        kernel = kernel / kernel.sum()

        # Add channel dimensions: (out_channels, in_channels, ...)
        if dims == 2:
            kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        else:
            kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)

        # Padding value (half of kernel size for symmetric padding)
        padding = kernel_size // 2

        return kernel, padding

    # Add channel dimension to noise
    if is_3d:
        noise_tensor = noise_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    else:
        noise_tensor = noise_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # Apply Gaussian smoothing with periodic boundaries
    kernel, padding = create_gaussian_kernel(correlation_length, len(shape))
    if is_3d:
        # For 3D: use CircularPad3d, then conv3d with padding=0
        # CircularPad3d padding: (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
        pad3d = torch.nn.CircularPad3d(padding)
        padded_noise = pad3d(noise_tensor)
        smoothed_noise = torch.nn.functional.conv3d(padded_noise, kernel, padding=0)
    else:
        # For 2D: use CircularPad2d, then conv2d with padding=0
        # CircularPad2d padding: (pad_left, pad_right, pad_top, pad_bottom) or single int
        pad2d = torch.nn.CircularPad2d(padding)
        padded_noise = pad2d(noise_tensor)
        smoothed_noise = torch.nn.functional.conv2d(padded_noise, kernel, padding=0)

    # Additional smoothing for better transitions
    if smoothing_sigma > 0:
        kernel_smooth, padding_smooth = create_gaussian_kernel(smoothing_sigma, len(shape))
        if is_3d:
            pad3d_smooth = torch.nn.CircularPad3d(padding_smooth)
            padded_noise = pad3d_smooth(smoothed_noise)
            smoothed_noise = torch.nn.functional.conv3d(padded_noise, kernel_smooth, padding=0)
        else:
            pad2d_smooth = torch.nn.CircularPad2d(padding_smooth)
            padded_noise = pad2d_smooth(smoothed_noise)
            smoothed_noise = torch.nn.functional.conv2d(padded_noise, kernel_smooth, padding=0)

    # Convert back to numpy and remove channel dimensions
    smoothed_noise = smoothed_noise.squeeze().numpy()

    # Convert to binary based on porosity threshold
    threshold = np.percentile(smoothed_noise, (1 - porosity) * 100)
    binary_field = smoothed_noise > threshold

    return binary_field
