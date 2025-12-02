"""
Periodic Watershed Segmentation

Marker-based watershed segmentation with support for periodic boundary conditions.
"""

from .periodic_watershed import watershed_periodic

periodic_watershed = watershed_periodic  # Alias for backwards compatibility

__all__ = ["periodic_watershed", "watershed_periodic"]
