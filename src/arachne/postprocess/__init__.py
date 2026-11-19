"""Post-processing with the full SPS model: spectra and derived quantities from a fit.

The emulator predicts photometry.  This package rebuilds the Synthesizer model
the emulator's training library was generated with, from the *simulator recipe*
stored in the checkpoint, and evaluates it on posterior draws:

- :mod:`arachne.postprocess.recipe`: the recipe format (pure Python).
- :mod:`arachne.postprocess.simulator`: :class:`Simulator` (needs Synthesizer and
  the SPS grid) and :func:`check_recipe`.
- :mod:`arachne.postprocess.bundle`: :func:`postprocess_bundle`, which adds
  region spectra and derived maps (surviving mass, SFR, ages) to a results bundle,
  and :func:`load_postprocess`.

See ``docs/guide/postprocessing.md``.
"""

from .recipe import recipe_from_checkpoint, recipe_from_library, recipe_hash

__all__ = ["recipe_from_checkpoint", "recipe_from_library", "recipe_hash"]
