#!/usr/bin/env python3
"""Benchmark emulator inference throughput (relevant to MCMC fitting cost).

Times jitted batched predict calls for the production ParrotEmulator and any
V2/lab checkpoints given on the command line.
"""

from __future__ import annotations

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np


def bench(fn, x, n_warm=3, n_rep=20):
    """Time ``fn(x)`` over ``n_rep`` repetitions after ``n_warm`` warm-up calls."""
    for _ in range(n_warm):
        fn(x).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(n_rep):
        fn(x).block_until_ready()
    return (time.perf_counter() - t0) / n_rep


def main(paths):
    """Benchmark inference speed for each emulator checkpoint in ``paths``."""
    from arachne.emulator.parrot_emulator_v2 import load_emulator

    rng = np.random.default_rng(0)
    # plausible raw parameter draws (12 params, v6 library ranges)
    n = 16384
    x = np.column_stack(
        [
            rng.uniform(0.01, 14, n),  # redshift
            rng.uniform(6, 12.5, n),  # log_mass
            rng.uniform(-0.3, 1.1, n),  # slope
            rng.uniform(0, 1, n),  # fesc_lya
            rng.uniform(0, 5, n),  # dust_bump
            rng.uniform(-4, -1.4, n),  # log10metallicity
            rng.uniform(0, 5, n),  # Av
            *[rng.standard_t(2, n) for _ in range(5)],  # logsfr ratios
        ]
    ).astype(np.float32)
    xj = jnp.array(x)

    for p in paths:
        emu = load_emulator(p)
        f = jax.jit(lambda x, e=emu: e.predict(x))
        dt = bench(f, xj)
        rate = n / dt / 1e6
        print(f"{p.split('/')[-1]:40s} {dt * 1e3:8.2f} ms / {n} -> {rate:6.2f} M SED/s")


if __name__ == "__main__":
    main(sys.argv[1:])
