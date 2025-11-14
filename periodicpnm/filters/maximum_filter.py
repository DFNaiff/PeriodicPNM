"""
A version of maximum filter that supports partially periodic boundaries,
through padding and cropping. We intend, in a future version, to use a more
efficient algorithm.
"""
import numpy as np
from scipy.ndimage import maximum_filter as scipy_maximum_filter

from periodicpnm.periodic_edt import periodic_edt


def maximum_filter(
    image,
    footprint,
    periodic_axes=None,
):
    if periodic_axes is None:
        periodic_axes = (False, ) * image.ndim
    elif isinstance(periodic_axes, bool):
        periodic_axes = (periodic_axes, ) * image.ndim
    elif len(periodic_axes) != image.ndim:
        raise ValueError(f"periodic_axes must be a bool or a sequence of {image.ndim} bools, got {len(periodic_axes)}")

    padding_sizes = [i // 2 for i in footprint.shape]

    periodic_padding = [[0 if not periodic else padding_sizes[i]] * 2 for i, periodic in enumerate(periodic_axes)]
    reflect_padding = [[0 if periodic else padding_sizes[i]] * 2 for i, periodic in enumerate(periodic_axes)]

    image = np.pad(image, periodic_padding, mode='wrap')
    image = np.pad(image, reflect_padding, mode='reflect')

    image = scipy_maximum_filter(image, footprint=footprint)
    image = image[tuple(slice(padding, -padding) for padding in padding_sizes)]
    return image


def find_peaks(
    dist,
    image,
    radius,
    periodic_axes=None,
):
    ndim = dist.ndim
    strel = ps_round(radius, ndim, smooth=False)
    dist_max = maximum_filter(dist + 2 * (~image), strel, periodic_axes=periodic_axes)
    peaks = (dist == dist_max) * image
    return peaks


def ps_round(r, ndim, smooth=True):
    r"""
    Taken from porespy.tools.ps_round.
    Creates round structuring element with the given radius and dimensionality

    Parameters
    ----------
    r : scalar
        The desired radius of the structuring element
    ndim : int
        The dimensionality of the element, either 2 or 3.
    smooth : boolean
        Indicates whether the faces of the sphere should have the little
        nibs (``True``) or not (``False``, default)

    Returns
    -------
    strel : ndarray
        A 3D numpy array of the structuring element

    Examples
    --------
    `Click here
    <https://porespy.org/examples/tools/reference/ps_round.html>`_
    to view online example.

    """
    rad = int(np.ceil(r))
    other = np.zeros([2*rad + 1 for i in range(ndim)], dtype=bool)
    other[tuple(rad for i in range(ndim))] = True
    if smooth:
        ball = periodic_edt(other) < r
    else:
        ball = periodic_edt(other) <= r
    return ball
