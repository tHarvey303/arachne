"""FFT-based differentiable PSF convolution with optional zero-padding and shifts."""

from __future__ import annotations

import copy

import jax.numpy as jnp
import numpy as np
from scipy.fft import next_fast_len

from arachne.data.psf import PSFModel
from arachne.utils.logging import setup_named_logger

logger = setup_named_logger(__name__)


def _fast_even_len(n: int) -> int:
    """Smallest *even* 5-smooth FFT length greater than or equal to ``n``.

    ``scipy.fft.next_fast_len`` returns the next length that factorises into
    small primes (2, 3, 5), for which the FFT is fastest.  Odd candidates are
    rejected and the search continues, because an even final axis keeps
    ``rfft2``/``irfft2`` on their cheapest code path and makes the Nyquist bin
    real (which matters for the sub-pixel shift phase ramp, see
    :meth:`PSFConvolver.__call__`).

    Args:
        n: Minimum required length.

    Returns:
        Even integer >= ``n`` whose only prime factors are 2, 3 and 5.
    """
    m = max(int(n), 2)
    while True:
        candidate = int(next_fast_len(m, real=True))
        if candidate % 2 == 0:
            return candidate
        m = candidate + 1


def _place_and_shift_kernels(kernels: np.ndarray, H: int, W: int) -> np.ndarray:
    """Zero-pad kernels to (H, W) with the kernel centre at (0, 0) after ifftshift.

    This mirrors :meth:`arachne.data.psf.PSFModel.pad_to_image_size` exactly,
    but operates on an arbitrary (typically padded) grid without going through
    ``PSFModel``.  The kernel centre pixel ``(kh // 2, kw // 2)`` is placed at
    ``(H // 2, W // 2)`` and ``ifftshift`` then moves it to ``(0, 0)``.

    Args:
        kernels: Kernel array of shape (N_bands, kh, kw).
        H: Target height.
        W: Target width.

    Returns:
        Array of shape (N_bands, H, W), already ifftshifted.

    Raises:
        ValueError: If a kernel is larger than the target grid.
    """
    _, kh, kw = kernels.shape
    if kh > H or kw > W:
        raise ValueError(f"PSF kernel ({kh}x{kw}) is larger than the FFT grid ({H}x{W}).")
    pad_top = H // 2 - kh // 2
    pad_left = W // 2 - kw // 2
    padded = np.pad(
        kernels,
        ((0, 0), (pad_top, H - kh - pad_top), (pad_left, W - kw - pad_left)),
    )
    return np.fft.ifftshift(padded, axes=(-2, -1))


class PSFConvolver:
    """Differentiable PSF convolution via FFT with pre-computed PSF spectra.

    The PSF FFTs are computed once at construction and frozen.  During
    inference, each call requires only two FFTs (forward + inverse) per band,
    making convolution cheap relative to emulator evaluation.

    Padding (``pad=True``, the default)
    -----------------------------------
    A bare FFT product is a *circular* convolution: flux within half a kernel
    width of one edge wraps around to the opposite edge.  With ``pad=True``
    the image is embedded in the top-left corner of a zero array of shape
    ``padded_shape`` — the smallest even 5-smooth size at least
    ``(H + kh - 1, W + kw - 1)`` — the kernel FFT is pre-computed on that same
    grid, and the result is cropped back to ``(H, W)``.  Because the padded
    grid is at least the full linear-convolution size, no wrap-around occurs
    and the crop equals ``scipy.signal.fftconvolve(image, kernel, mode="same")``
    (for odd-sized kernels; even kernels follow this module's
    ``centre = kh // 2`` convention).

    ``pad=False`` reproduces the historical circular behaviour bit-for-bit.

    The ifftshift convention
    ------------------------
    ifftshift is applied to the padded PSF *before* computing its FFT.  This
    moves the PSF peak from the grid centre (where it sits naturally after
    padding) to pixel (0, 0) — the convention required so that
    ``irfft2(rfft2(image) * rfft2(psf))`` yields a convolution without any
    spatial shift artefact.

    **Missing ifftshift is the most common PSF convolution bug**: the sampler
    will silently compensate by shifting the spatial model, producing
    incorrectly inferred component positions.  The test ``test_psf_centering``
    guards against this.

    Sub-pixel shifts
    ----------------
    ``__call__(image, shifts=...)`` applies a per-band translation as a Fourier
    phase ramp folded into the PSF spectrum.  It is exact (band-limited sinc
    interpolation), differentiable, and essentially free — no extra FFT.

    Single-band and arbitrary grids
    -------------------------------
    Nothing here is tied to a particular number of bands or to a square image:
    ``n_bands`` may be 1, ``image_shape`` may be any ``(H, W)``, and the kernel
    size ``(kh, kw)`` is independent of the image size (with ``pad=True`` the
    kernel may even be *larger* than the image, since the FFT grid is sized
    from ``H + kh - 1``; with ``pad=False`` a kernel larger than the image
    raises ``ValueError``).  Use :meth:`from_kernels` when the PSF is a plain
    array rather than a :class:`~arachne.data.psf.PSFModel` — for instance when
    each band is convolved on its own pixel grid with its own kernel.

    Attributes:
        psf_ffts: Pre-computed PSF FFTs of shape (N_bands, Hp, Wp//2+1), complex.
        image_shape: Spatial image dimensions (H, W) of the science frame; also
            the shape ``__call__`` accepts and returns (per band).
        padded_shape: The ``(Hp, Wp)`` FFT grid the convolution is actually
            evaluated on.  With ``pad=False`` it is exactly ``image_shape``.
            With ``pad=True`` it is ``(_fast_even_len(H + kh - 1),
            _fast_even_len(W + kw - 1))``: the smallest even 5-smooth size at
            least the full linear-convolution length, so the zero-padded image
            and the kernel cannot wrap into each other.  ``Hp >= H`` and
            ``Wp >= W`` always hold; the image is placed at the top-left corner
            of the padded grid and the result is cropped back to ``[:H, :W]``.
            It is a plain Python tuple of ints, safe to read at trace time.
        pad: Whether zero-padding (linear convolution) is enabled.
        kernel_shape: Native PSF kernel dimensions (kh, kw).
        n_bands: Number of photometric bands (may be 1).
    """

    def __init__(
        self,
        psf_model: PSFModel,
        image_shape: tuple[int, int],
        pad: bool = True,
    ) -> None:
        """Pre-compute PSF FFTs for the given image shape.

        Args:
            psf_model: PSFModel containing per-band kernel arrays.
            image_shape: (H, W) spatial dimensions of the science image.
            pad: When True (default) convolve on a zero-padded grid so the
                result is a linear (non-wrapping) convolution truncated to the
                frame.  When False the historical circular convolution on the
                (H, W) grid is used.
        """
        H, W = image_shape
        self.image_shape = (H, W)
        self.n_bands = psf_model.n_bands
        self.pad = bool(pad)

        raw_kernels = np.asarray(psf_model.kernels)
        self.kernel_shape = (int(raw_kernels.shape[-2]), int(raw_kernels.shape[-1]))
        kh, kw = self.kernel_shape

        if self.pad:
            Hp = _fast_even_len(H + kh - 1)
            Wp = _fast_even_len(W + kw - 1)
            kernels = _place_and_shift_kernels(raw_kernels, Hp, Wp)
        else:
            Hp, Wp = H, W
            # Delegate to PSFModel so the unpadded path stays bit-for-bit
            # identical to the historical implementation.
            kernels = np.asarray(psf_model.pad_to_image_size(H, W).kernels)
        self.padded_shape = (int(Hp), int(Wp))

        # Normalise each kernel to sum to 1 (flux conservation)
        kernel_sums = kernels.sum(axis=(-2, -1), keepdims=True)
        kernel_sums = np.where(kernel_sums == 0, 1.0, kernel_sums)
        kernels = kernels / kernel_sums

        # Pre-compute rfft2 — the PSF FFT peak is at (0,0) after ifftshift
        psf_ffts_np = np.fft.rfft2(kernels)  # (N_bands, Hp, Wp//2+1) complex
        self.psf_ffts = jnp.array(psf_ffts_np)

        # Frequency grids in cycles per pixel, for the sub-pixel shift ramp.
        self._freq_y = jnp.array(np.fft.fftfreq(self.padded_shape[0]), dtype=jnp.float32)
        self._freq_x = jnp.array(np.fft.rfftfreq(self.padded_shape[1]), dtype=jnp.float32)

        logger.info(
            f"PSFConvolver initialised: {self.n_bands} bands, image shape {self.image_shape}, "
            f"pad={self.pad}, FFT grid {self.padded_shape}, PSF FFT shape {self.psf_ffts.shape}"
        )

    @classmethod
    def from_kernels(
        cls,
        kernels: np.ndarray | jnp.ndarray,
        image_shape: tuple[int, int],
        pad: bool = True,
        band_names: list[str] | None = None,
    ) -> "PSFConvolver":
        """Build a convolver from raw kernel arrays, without a :class:`PSFModel`.

        Convenience entry point for code that holds a PSF as a plain array on
        one band's own pixel grid (e.g. a multi-resolution fit, where each band
        is convolved on its native grid with its own kernel).  The kernel may
        be any size, independent of ``image_shape``; with ``pad=True`` it may
        even be larger than the image, because the FFT grid is sized from
        ``H + kh - 1``.

        Args:
            kernels: Kernel array of shape (N_bands, kh, kw), or (kh, kw) for a
                single band (promoted to (1, kh, kw)).  Kernels are normalised
                to unit sum internally.
            image_shape: (H, W) of the science image on this grid.
            pad: Use the zero-padded (linear) convolution.  See the class
                docstring.
            band_names: Optional names, only used for logging/labelling.
                Defaults to ``band_0 ... band_{N-1}``.

        Returns:
            Configured ``PSFConvolver``.

        Raises:
            ValueError: If ``kernels`` is not 2-D or 3-D.
        """
        arr = np.asarray(kernels)
        if arr.ndim == 2:
            arr = arr[None, :, :]
        elif arr.ndim != 3:
            raise ValueError(f"kernels must be (N_bands, kh, kw) or (kh, kw), got {arr.shape}")
        if band_names is None:
            band_names = [f"band_{i}" for i in range(arr.shape[0])]
        return cls(
            PSFModel(kernels=arr, band_names=band_names),
            image_shape=image_shape,
            pad=pad,
        )

    def _shift_phase(self, shifts: jnp.ndarray) -> jnp.ndarray:
        """Fourier phase ramp implementing a per-band translation.

        Args:
            shifts: Array of shape (N_bands, 2) holding ``(dy, dx)`` in pixels.
                Positive ``dy`` moves the image towards higher row indices
                (down), positive ``dx`` towards higher column indices (right).

        Returns:
            Complex array of shape (N_bands, Hp, Wp//2+1).
        """
        shifts = jnp.asarray(shifts)
        dy = shifts[:, 0][:, None, None]
        dx = shifts[:, 1][:, None, None]
        phase = self._freq_y[None, :, None] * dy + self._freq_x[None, None, :] * dx
        return jnp.exp(-2j * jnp.pi * phase)

    def with_psf_ffts(self, psf_ffts: jnp.ndarray) -> "PSFConvolver":
        """Return a shallow copy using different pre-computed PSF spectra.

        The spectra must have been computed on this convolver's FFT grid (same
        shape as ``self.psf_ffts``); ``psf_ffts`` may be a tracer, which is how
        a batched program swaps in each galaxy's PSF.

        Args:
            psf_ffts: Array with the same shape as ``self.psf_ffts``.

        Returns:
            A copy sharing all static configuration.

        Raises:
            ValueError: On a shape mismatch.
        """
        if tuple(psf_ffts.shape) != tuple(self.psf_ffts.shape):
            raise ValueError(
                f"psf_ffts must have shape {tuple(self.psf_ffts.shape)}, "
                f"got {tuple(psf_ffts.shape)}."
            )
        new = copy.copy(self)
        new.psf_ffts = psf_ffts
        return new

    def __call__(self, image: jnp.ndarray, shifts: jnp.ndarray | None = None) -> jnp.ndarray:
        """Convolve a multi-band image with the per-band PSFs.

        This function is fully differentiable with ``jax.grad``, in both the
        image and the shifts.

        Args:
            image: Multi-band image array of shape (N_bands, H, W).
            shifts: Optional per-band sub-pixel offsets of shape (N_bands, 2),
                ``(dy, dx)`` in pixels.  Positive ``dy`` moves the model down
                (towards higher row index), positive ``dx`` to the right.
                ``None`` (default) applies no shift and costs nothing.

        Returns:
            PSF-convolved image of shape (N_bands, H, W).
        """
        H, W = self.image_shape
        Hp, Wp = self.padded_shape

        if self.pad:
            image = jnp.pad(image, ((0, 0), (0, Hp - H), (0, Wp - W)))

        image_fft = jnp.fft.rfft2(image)  # (N_bands, Hp, Wp//2+1) complex
        psf_fft = self.psf_ffts
        if shifts is not None:
            psf_fft = psf_fft * self._shift_phase(shifts)

        convolved = jnp.fft.irfft2(image_fft * psf_fft, s=(Hp, Wp))
        if self.pad:
            convolved = convolved[:, :H, :W]
        return convolved  # (N_bands, H, W)
