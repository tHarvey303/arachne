"""FITS helpers and HDF5 result serialisation utilities."""

from pathlib import Path
from typing import Any

import h5py
import jax.numpy as jnp
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from arachne.utils.logging import setup_named_logger

logger = setup_named_logger(__name__)


def load_fits_image(path: str | Path) -> tuple[np.ndarray, fits.Header]:
    """Load the primary extension of a FITS file.

    Args:
        path: Path to the FITS file.

    Returns:
        Tuple of (data array, FITS header).
    """
    with fits.open(path) as hdul:
        data = hdul[0].data.astype(np.float32)
        header = hdul[0].header
    return data, header


def load_fits_wcs(path: str | Path) -> WCS:
    """Extract WCS from a FITS file header.

    Args:
        path: Path to the FITS file.

    Returns:
        Astropy WCS object.
    """
    with fits.open(path) as hdul:
        header = hdul[0].header
    return WCS(header)


def save_array_to_hdf5(group: h5py.Group, name: str, array: np.ndarray | jnp.ndarray) -> None:
    """Save an array to an HDF5 group.

    Args:
        group: Open HDF5 group.
        name: Dataset name within the group.
        array: Array to save (numpy or JAX).
    """
    if hasattr(array, "__jax_array__") or isinstance(array, jnp.ndarray):
        array = np.asarray(array)
    group.create_dataset(name, data=array, compression="gzip")


def load_array_from_hdf5(group: h5py.Group, name: str) -> np.ndarray:
    """Load an array from an HDF5 group.

    Args:
        group: Open HDF5 group.
        name: Dataset name within the group.

    Returns:
        Numpy array loaded from HDF5.
    """
    return group[name][()]


def save_attrs_to_hdf5(group: h5py.Group, attrs: dict[str, Any]) -> None:
    """Save a dictionary of scalar attributes to an HDF5 group.

    Args:
        group: Open HDF5 group.
        attrs: Dictionary of attribute name → value pairs.
    """
    for key, value in attrs.items():
        if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
            group.attrs[key] = np.bytes_(value)
        else:
            group.attrs[key] = value
