"""Tests for the static plots in ``arachne.viz`` (Agg backend, synthetic bundles)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from arachne import viz  # noqa: E402
from arachne.results import (  # noqa: E402
    Region,
    ResultsBundle,
    component_seds,
    make_synthetic_bundle,
    make_synthetic_content,
    region_sed,
    total_sed,
)
from arachne.viz import cli  # noqa: E402

FLAVOURS = ("field", "additive", "degraded")


@pytest.fixture(scope="module")
def bundles():
    """In-memory bundles of the three synthetic flavours."""
    return {f: ResultsBundle.from_content(make_synthetic_content(f)) for f in FLAVOURS}


def _render(fig: Figure) -> None:
    fig.canvas.draw()


def _main_axes(fig: Figure) -> list:
    """Visible axes that are not colourbars."""
    return [a for a in fig.axes if a.get_label() != "<colorbar>" and a.axison]


BUNDLE_FUNCS = {
    "plot_total_sed": lambda b: viz.plot_total_sed(b),
    "plot_region_seds": lambda b: viz.plot_region_seds(b),
    "plot_region_seds_overlay": lambda b: viz.plot_region_seds(b, layout="overlay"),
    "plot_component_seds": lambda b: viz.plot_component_seds(b),
    "plot_param_maps": lambda b: viz.plot_param_maps(b),
    "plot_regions": lambda b: viz.plot_regions(b),
    "plot_band_mosaic": lambda b: viz.plot_band_mosaic(b),
    "plot_rgb": lambda b: viz.plot_rgb(b),
    "plot_chi_histograms": lambda b: viz.plot_chi_histograms(b),
    "plot_chi2_per_band": lambda b: viz.plot_chi2_per_band(b),
    "plot_summary_page": lambda b: viz.plot_summary_page(b),
}


@pytest.mark.parametrize("flavour", FLAVOURS)
@pytest.mark.parametrize("name", list(BUNDLE_FUNCS))
def test_every_function_returns_figure(bundles, flavour, name):
    """Every plotting function returns a drawable Figure for every flavour."""
    fig = BUNDLE_FUNCS[name](bundles[flavour])
    assert isinstance(fig, Figure)
    _render(fig)


def test_plot_sed_axes_and_twin(bundles):
    """Two residual panels give 3 axes plus the band-name twin on the top panel."""
    sed = total_sed(bundles["field"])
    fig = viz.plot_sed(sed)
    assert len(fig.axes) == 3
    top = fig.axes[0]
    assert len(top.child_axes) == 1  # secondary (band-name) x axis
    _render(fig)
    labels = [t.get_text() for t in top.child_axes[0].get_xticklabels()]
    assert labels == ["F115W", "F200W", "F444W"]
    assert top.get_xscale() == "log"

    assert len(viz.plot_sed(sed, residuals=("sigma",)).axes) == 2
    assert len(viz.plot_sed(sed, residuals=()).axes) == 1
    with pytest.raises(ValueError):
        viz.plot_sed(sed, residuals=("bogus",))


def test_sigma_panel_matches_region_chi(bundles):
    """The sigma residual panel plots exactly RegionSED.chi()."""
    b = bundles["field"]
    sed = region_sed(b, b.regions["core"])
    fig = viz.plot_sed(sed, residuals=("sigma",))
    rax = fig.axes[1]
    pts = np.concatenate([c.get_offsets() for c in rax.collections if c.get_offsets().size])
    np.testing.assert_allclose(pts[:, 0], sed.pivot_um)
    np.testing.assert_array_equal(pts[:, 1], sed.chi())


def test_frac_panel_matches_frac_residual(bundles):
    """The fractional panel shows 100 * RegionSED.frac_residual()."""
    sed = total_sed(bundles["field"])
    fig = viz.plot_sed(sed, residuals=("frac",))
    rax = fig.axes[1]
    scat = [
        c for c in rax.collections if c.get_offsets().size and not isinstance(c, PolyCollection)
    ]
    ys = np.asarray(scat[-1].get_offsets())[:, 1]
    np.testing.assert_allclose(ys, 100 * sed.frac_residual())


def _n_fills(ax) -> int:
    return sum(isinstance(c, PolyCollection) for c in ax.collections)


def test_sed_degrades_without_draws(bundles):
    """No draws: no 16-84 bands and the model is labelled MAP; draws: bands present."""
    deg = viz.plot_sed(total_sed(bundles["degraded"]), show_intrinsic=True)
    top = deg.axes[0]
    assert _n_fills(top) == 0
    labels = top.get_legend_handles_labels()[1]
    assert "model (MAP)" in labels
    assert not any(lab.startswith("intrinsic") for lab in labels)  # no intrinsic model
    assert _n_fills(deg.axes[2]) == 0  # frac panel without model band

    full = viz.plot_sed(total_sed(bundles["field"]), show_intrinsic=True)
    labels = full.axes[0].get_legend_handles_labels()[1]
    assert _n_fills(full.axes[0]) == 2  # convolved + intrinsic
    assert any(lab.startswith("intrinsic (unconvolved") for lab in labels)
    assert "truth (convolved, noise-free)" in labels


def test_sed_without_truth():
    """Bundles without truth draw no truth markers."""
    b = ResultsBundle.from_content(make_synthetic_content("field", with_truth=False))
    fig = viz.plot_sed(total_sed(b), show_intrinsic=True)
    assert not any("truth" in lab for lab in fig.axes[0].get_legend_handles_labels()[1])
    fig = viz.plot_param_maps(b)
    assert len([a for a in fig.axes if a.images]) == 2 * 3  # p50, std, information
    _render(fig)


def test_component_sed_in_plot_sed(bundles):
    """Component SEDs (intrinsic only, no obs) plot without residual panels."""
    seds = component_seds(bundles["additive"])
    fig = viz.plot_sed(seds[0])
    assert len(fig.axes) == 1
    _render(fig)


def test_component_seds_figure(bundles):
    """Additive: intrinsic + observed-frame panels; field: a note figure."""
    fig = viz.plot_component_seds(bundles["additive"])
    assert len(fig.axes) == 2
    labels = fig.axes[0].get_legend_handles_labels()[1]
    assert sum(lab.startswith("comp ") for lab in labels) == 2
    note = viz.plot_component_seds(bundles["field"])
    assert len(note.axes) == 1 and not note.axes[0].axison


def test_region_seds_grid_and_overlay(bundles):
    """Grid: one SED (3 axes) per region; overlay: 2 axes; bad layout raises."""
    b = bundles["field"]
    fig = viz.plot_region_seds(b)
    assert len(fig.axes) == 3 * len(b.regions)
    fig = viz.plot_region_seds(b, ["core"], residuals=("sigma",))
    assert len(fig.axes) == 2
    mask = np.zeros(b.image_shape, bool)
    mask[4:8, 4:8] = True
    fig = viz.plot_region_seds(b, {"box": mask, "c": Region.circle(b.image_shape, 8, 8, 2)})
    assert len(fig.axes) == 6
    assert len(viz.plot_region_seds(b, layout="overlay").axes) == 2
    with pytest.raises(ValueError):
        viz.plot_region_seds(b, layout="nope")
    with pytest.raises(KeyError):
        viz.plot_region_seds(b, ["missing"])


@pytest.mark.parametrize(
    "flavour, n_images",
    [("field", 2 * 5), ("degraded", 2 * 5), ("additive", 0)],
)
def test_param_maps_panels(bundles, flavour, n_images):
    """Default stats: truth, p50, std, z, information for each map parameter."""
    fig = viz.plot_param_maps(bundles[flavour])
    assert len([a for a in fig.axes if a.images]) == n_images
    if n_images:
        titles = [a.get_title() for a in fig.axes[:5]]
        assert titles[0] == "truth" and titles[1] == "median"


def test_param_maps_explicit(bundles):
    """Explicit names/stats; unavailable statistics (no MAP fallback) show a note panel."""
    fig = viz.plot_param_maps(bundles["degraded"], names=["Av"], stats=["map", "p84"])
    assert len([a for a in fig.axes if a.images]) == 1


def test_band_mosaic_panels(bundles):
    """Rows = bands, columns = available kinds (intrinsic dropped when absent)."""
    b = bundles["field"]
    fig = viz.plot_band_mosaic(b)
    assert len([a for a in fig.axes if a.images]) == 3 * 4
    fig = viz.plot_band_mosaic(b, bands=[0], kinds=("data", "intrinsic", "truth_model"))
    assert len([a for a in fig.axes if a.images]) == 3
    deg = viz.plot_band_mosaic(bundles["degraded"], kinds=("data", "model", "intrinsic"))
    axes = [a for a in deg.axes if a.images]
    assert len(axes) == 3 * 2
    assert axes[1].get_title() == "model (MAP)"


def test_rgb_and_default_bands(bundles):
    """Default RGB = reddest/middle/bluest; one panel per available kind."""
    b = bundles["field"]
    assert viz.images.default_rgb_bands(b) == (2, 1, 0)
    fig = viz.plot_rgb(b, kinds=("data", "model", "residual", "truth_model"))
    assert len(fig.axes) == 4
    img = fig.axes[0].images[0].get_array()
    assert img.shape == (*b.image_shape, 3)
    assert float(np.min(img)) >= 0.0 and float(np.max(img)) <= 1.0
    assert len(viz.plot_rgb(bundles["degraded"], kinds=("data", "intrinsic")).axes) == 1


def test_diagnostics(bundles):
    """One histogram per band; chi2 per band plots every band."""
    b = bundles["field"]
    assert len(_main_axes(viz.plot_chi_histograms(b))) == b.n_bands
    fig = viz.plot_chi2_per_band(b)
    pts = np.concatenate([c.get_offsets() for c in fig.axes[0].collections])
    np.testing.assert_allclose(np.sort(pts[:, 1]), np.sort(b.chi2_red_per_band()))


def test_axes_kw(bundles):
    """Functions with ``ax=`` draw into the given axes."""
    fig = Figure()
    ax = fig.add_subplot()
    out = viz.plot_chi2_per_band(bundles["field"], ax=ax)
    assert out is fig
    fig2 = Figure()
    ax2 = fig2.add_subplot()
    assert viz.plot_regions(bundles["field"], ax=ax2) is fig2


def test_band_colors():
    """Colours run blue to red with wavelength; unknown pivots are grey."""
    cols = viz.band_colors([4.4, 1.15, np.nan, 2.0])
    assert cols[2] == (0.5, 0.5, 0.5, 1.0)
    blue, red = cols[1], cols[0]
    assert blue[2] > blue[0] and red[0] > red[2]
    assert viz.band_colors([2.0])[0] != (0.5, 0.5, 0.5, 1.0)
    assert viz.region_color(0) == viz.region_color(len(viz.style.REGION_COLORS))


EXPECTED = {
    "field": [
        "summary", "total_sed", "region_seds", "param_maps", "regions", "band_mosaic", "rgb",
        "chi_histograms", "chi2_per_band",
    ],
    "additive": [
        "summary", "total_sed", "region_seds", "component_seds", "regions", "band_mosaic", "rgb",
        "chi_histograms", "chi2_per_band",
    ],
    "degraded": [
        "summary", "total_sed", "region_seds", "param_maps", "regions", "band_mosaic", "rgb",
        "chi_histograms", "chi2_per_band",
    ],
}  # fmt: skip


@pytest.mark.parametrize("flavour", FLAVOURS)
def test_save_all_file_list(tmp_path, flavour):
    """save_all writes the expected files from a bundle on disk (path input)."""
    path = make_synthetic_bundle(tmp_path / f"{flavour}.h5", flavour)
    with _no_warnings():
        paths = viz.save_all(str(path), tmp_path / "figs")
    assert [p.stem for p in paths] == EXPECTED[flavour]
    assert all(p.suffix == ".png" and p.stat().st_size > 0 for p in paths)


class _no_warnings:
    """Fail on any warning raised inside the block."""

    def __enter__(self):
        import warnings

        self._cm = warnings.catch_warnings()
        self._cm.__enter__()
        warnings.simplefilter("error")
        return self

    def __exit__(self, *exc):
        return self._cm.__exit__(*exc)


def test_save_all_unknown_region_warns(tmp_path, bundles):
    """Unknown region names are skipped with a warning."""
    with pytest.warns(UserWarning, match="not in bundle"):
        paths = viz.save_all(bundles["field"], tmp_path, regions=["core", "nope"])
    assert (tmp_path / "region_seds.png") in paths


def test_cli(tmp_path, capsys):
    """``arachne-plots`` writes the figures in the requested format."""
    path = make_synthetic_bundle(tmp_path / "b.h5", "field")
    out = tmp_path / "out"
    rc = cli.main([str(path), "--outdir", str(out), "--format", "pdf", "--regions", "core"])
    assert rc == 0
    written = sorted(p.name for p in out.iterdir())
    assert written == sorted(f"{n}.pdf" for n in EXPECTED["field"])
    assert "wrote 9 figures" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["--help"])
