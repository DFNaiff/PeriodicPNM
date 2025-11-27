import numpy as np
from scipy.ndimage import label

from periodicpnm.periodic_edt import periodic_edt
from periodicpnm.filters import gaussian_filter
from periodicpnm.filters import find_peaks
from periodicpnm.filters import trim_nearby_peaks, trim_saddle_points
from periodicpnm.watershed import watershed_periodic


def periodic_snow(
    im,
    periodic_axes=(True, True),
    filter_sigma=0.4,
    saddle_maxiter=20,
    nearby_f=1.0,
    peak_maximum_radius=4,
    label_strel=3,
    watershed_eps=1e-6
):
    """
    Periodic SNOW algorithm.

    Parameters
    ----------
    im : ndarray
        Binary image. 1 are considered features (pores), 0 solids
    periodic_axes : tuple, optional
        Periodic axes.
    """
    ndim = im.ndim
    dist = periodic_edt(
        im,
        periodic_axes=periodic_axes,
        feature=0
    )
    dist_blurred = gaussian_filter(dist, filter_sigma, periodic_axes=periodic_axes)
    peaks = find_peaks(dist_blurred, im, radius=peak_maximum_radius, periodic_axes=periodic_axes)
    peaks = trim_saddle_points(peaks, dist, periodic_axes=periodic_axes, maxiter=saddle_maxiter)
    peaks = trim_nearby_peaks(peaks, dist, periodic_axes=periodic_axes, f=nearby_f)
    peaks, N = label(peaks > 0, structure=np.ones((3,)*ndim))
    partitioning = watershed_periodic(-dist + watershed_eps, peaks, periodic_axes=periodic_axes)
    partitioning = partitioning * im
    return_dict = {
        'im': im,
        'dist': dist,
        'peaks': peaks,
        'partitioning': partitioning
    }
    return return_dict
