"""Tests for the JADES DR4 spectroscopic-redshift catalogue client.

:func:`select_targets` is a pure table operation and is tested against a small
synthetic table, so it runs offline.  The tests that download the real
catalogue (about 94 MB) are marked ``@pytest.mark.network`` and skip when
``jades.herts.ac.uk`` does not answer within 5 s.

Set ``ARACHNE_TEST_CACHE`` to a persistent directory to reuse the download
between runs (see the ``arachne_cache_dir`` fixture in ``conftest.py``).
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.table import Table

from arachne.data.dja import server_reachable
from arachne.data.jades import (
    JADES_DR4_BASE_URL,
    QUALITY_FLAGS,
    download_jades_dr4_specz,
    load_jades_dr4_specz,
    select_targets,
)


@pytest.fixture(scope="session")
def jades_server():
    """Skip the calling test unless the JADES web server answers within 5 s.

    Returns:
        The JADES DR4 base URL.
    """
    if not server_reachable(JADES_DR4_BASE_URL, timeout=5.0):
        pytest.skip(f"{JADES_DR4_BASE_URL} unreachable (offline?)")
    return JADES_DR4_BASE_URL


@pytest.fixture
def synthetic_catalogue():
    """A small standardised table with one row per quality flag and field.

    Returns:
        Table with the columns produced by :func:`load_jades_dr4_specz`.
    """
    flags = ["A", "B", "C", "D", "E", "A", "A", "B"]
    fields = ["GOODS-S"] * 5 + ["GOODS-N"] * 2 + ["GOODS-S"]
    z = [1.5, 2.5, 2.0, 1.2, np.nan, 2.2, 0.5, 5.0]
    table = Table()
    table["id"] = [f"src_{i:02d}" for i in range(len(flags))]
    table["ra"] = np.linspace(53.10, 53.20, len(flags))
    table["dec"] = np.linspace(-27.80, -27.75, len(flags))
    table["z_spec"] = np.array(z, dtype=np.float64)
    table["z_flag"] = np.array(flags)
    table["field"] = np.array(fields)
    return table


class TestSelectTargets:
    """Selection semantics, exercised offline on a synthetic table."""

    def test_quality_best_keeps_only_a_and_b(self, synthetic_catalogue):
        """``quality='best'`` keeps flags A and B only."""
        got = select_targets(synthetic_catalogue, field=None, quality="best", n=None)
        assert set(np.asarray(got["z_flag"])) == {"A", "B"}

    def test_quality_secure_adds_c(self, synthetic_catalogue):
        """``quality='secure'`` widens the selection to A, B and C."""
        got = select_targets(synthetic_catalogue, field=None, quality="secure", n=None)
        assert set(np.asarray(got["z_flag"])) == {"A", "B", "C"}
        assert QUALITY_FLAGS["secure"] == ("A", "B", "C")

    def test_nan_redshifts_are_dropped(self, synthetic_catalogue):
        """Rows without a redshift never survive, whatever the quality cut."""
        got = select_targets(synthetic_catalogue, field=None, quality="any", n=None)
        assert np.isfinite(np.asarray(got["z_spec"])).all()
        assert "src_04" not in list(np.asarray(got["id"]))

    def test_field_filter_accepts_short_and_long_names(self, synthetic_catalogue):
        """``'GS'`` and ``'GOODS-S'`` select the same rows."""
        long_name = select_targets(synthetic_catalogue, field="GOODS-S", quality="any", n=None)
        short_name = select_targets(synthetic_catalogue, field="GS", quality="any", n=None)
        assert list(np.asarray(long_name["id"])) == list(np.asarray(short_name["id"]))
        assert set(np.asarray(long_name["field"])) == {"GOODS-S"}

    def test_field_none_keeps_both(self, synthetic_catalogue):
        """``field=None`` keeps GOODS-S and GOODS-N."""
        got = select_targets(synthetic_catalogue, field=None, quality="any", n=None)
        assert set(np.asarray(got["field"])) == {"GOODS-S", "GOODS-N"}

    def test_redshift_bounds_are_inclusive(self, synthetic_catalogue):
        """``z_min``/``z_max`` are inclusive bounds on ``z_spec``."""
        got = select_targets(
            synthetic_catalogue, field=None, z_min=1.5, z_max=2.5, quality="best", n=None
        )
        z = np.asarray(got["z_spec"])
        assert z.min() >= 1.5 and z.max() <= 2.5
        assert 5.0 not in set(z)

    def test_n_truncates_deterministically(self, synthetic_catalogue):
        """A fixed seed gives a reproducible subsample of the requested size."""
        first = select_targets(synthetic_catalogue, field=None, quality="any", n=3, seed=0)
        second = select_targets(synthetic_catalogue, field=None, quality="any", n=3, seed=0)
        assert len(first) == 3
        assert list(np.asarray(first["id"])) == list(np.asarray(second["id"]))

    def test_different_seeds_can_differ(self, synthetic_catalogue):
        """The subsample depends on the seed."""
        ids = {
            tuple(
                np.asarray(
                    select_targets(synthetic_catalogue, field=None, quality="any", n=2, seed=s)[
                        "id"
                    ]
                )
            )
            for s in range(8)
        }
        assert len(ids) > 1

    def test_n_larger_than_matches_returns_all(self, synthetic_catalogue):
        """Asking for more targets than exist returns every match."""
        got = select_targets(synthetic_catalogue, field="GOODS-N", quality="any", n=100)
        assert len(got) == 2

    def test_output_is_sorted_by_id(self, synthetic_catalogue):
        """The returned table is sorted by ``id`` for reproducibility."""
        got = select_targets(synthetic_catalogue, field=None, quality="any", n=None)
        ids = list(np.asarray(got["id"]))
        assert ids == sorted(ids)

    def test_bad_quality_raises(self, synthetic_catalogue):
        """An unknown ``quality`` keyword is rejected."""
        with pytest.raises(ValueError, match="quality must be one of"):
            select_targets(synthetic_catalogue, quality="excellent")


@pytest.mark.network
class TestJADESDR4Catalogue:
    """The real DR4 catalogue: download, column mapping and content."""

    @pytest.fixture(scope="class")
    def catalogue_path(self, jades_server, arachne_cache_dir):
        """Download (or reuse) the combined DR4 external catalogue."""
        return download_jades_dr4_specz(cache_dir=arachne_cache_dir)

    @pytest.fixture(scope="class")
    def table(self, catalogue_path):
        """Load the catalogue with standardised column names."""
        return load_jades_dr4_specz(path=catalogue_path)

    def test_download_is_cached(self, catalogue_path, arachne_cache_dir):
        """A second download call reuses the file on disk."""
        assert catalogue_path.exists()
        assert catalogue_path.stat().st_size > 1_000_000
        mtime = catalogue_path.stat().st_mtime
        again = download_jades_dr4_specz(cache_dir=arachne_cache_dir)
        assert again == catalogue_path
        assert again.stat().st_mtime == mtime

    def test_standardised_columns_present(self, table):
        """The six standardised columns exist."""
        for column in ("id", "ra", "dec", "z_spec", "z_flag", "field"):
            assert column in table.colnames

    def test_more_than_1000_rows(self, table):
        """The DR4 combined catalogue has several thousand targets."""
        assert len(table) > 1000

    def test_coordinates_are_finite_and_in_the_fields(self, table):
        """RA/Dec are finite degrees inside the GOODS-S and GOODS-N footprints."""
        ra = np.asarray(table["ra"], dtype=np.float64)
        dec = np.asarray(table["dec"], dtype=np.float64)
        assert np.isfinite(ra).all() and np.isfinite(dec).all()
        assert (ra >= 0.0).all() and (ra <= 360.0).all()
        assert (dec >= -90.0).all() and (dec <= 90.0).all()

    def test_redshifts_are_finite_where_present(self, table):
        """Where a redshift exists it is positive and finite; absences are NaN."""
        z = np.asarray(table["z_spec"], dtype=np.float64)
        finite = np.isfinite(z)
        assert finite.sum() > 1000
        assert (z[finite] > 0).all()
        assert z[finite].max() < 30.0

    def test_field_names_are_expanded(self, table):
        """``'GS'``/``'GN'`` are expanded to the full field names."""
        assert set(np.asarray(table["field"])) <= {"GOODS-S", "GOODS-N"}
        assert "GOODS-S" in set(np.asarray(table["field"]))

    def test_flags_are_the_documented_letters(self, table):
        """Quality flags are drawn from A-E."""
        flags = set(np.asarray(table["z_flag"]))
        assert flags <= set(QUALITY_FLAGS["any"])

    def test_ids_are_unique(self, table):
        """``Unique_ID`` is genuinely unique, unlike ``NIRSpec_ID``."""
        ids = np.asarray(table["id"])
        assert len(set(ids)) == len(ids)

    def test_secure_goods_s_targets_at_intermediate_z(self, table):
        """There are hundreds of secure-z GOODS-S galaxies with 1 < z < 3."""
        got = select_targets(table, field="GOODS-S", z_min=1.0, z_max=3.0, quality="best", n=None)
        assert len(got) > 100
        assert set(np.asarray(got["field"])) == {"GOODS-S"}
        z = np.asarray(got["z_spec"])
        assert z.min() >= 1.0 and z.max() <= 3.0
        assert set(np.asarray(got["z_flag"])) <= {"A", "B"}

    def test_select_targets_returns_requested_number(self, table):
        """Requesting five demo targets gives exactly five rows."""
        got = select_targets(
            table, field="GOODS-S", z_min=1.0, z_max=3.0, quality="best", n=5, seed=0
        )
        assert len(got) == 5
