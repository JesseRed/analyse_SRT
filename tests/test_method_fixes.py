"""Tests for the PELT penalty calibration and the HSMM priors."""

from __future__ import annotations

import numpy as np
import pytest


def _blocks(rng, n, profile, sd=0.2):
    return {b + 1: np.exp(np.log(0.5) + np.asarray(profile) + rng.normal(0, sd, 7)) for b in range(n)}


def test_pelt_bootstrap_null_is_centred_noise():
    pelt = pytest.importorskip("src.chunking.methods.change_point_pelt")
    rng = np.random.default_rng(0)
    spike = [0, 0, 0, 0.5, 0, 0, 0]
    target = _blocks(rng, 30, spike)
    yellow = {b + 100: v for b, v in _blocks(rng, 30, [0] * 7).items()}
    yellow = {b - 100 + 30: v for b, v in yellow.items()}  # blocks 31..60
    specs = pelt._window_specs(1, 60, 60, 10)
    null = pelt._build_bootstrap_null_signals(
        target, yellow, specs, sequence_type="blue", min_blocks=6, n_per_window=50, random_state=1
    )
    arr = np.vstack(null)
    assert arr.shape == (50, 7)
    assert np.allclose(arr.mean(axis=0), 0.0, atol=1e-12)   # systematic profile removed
    assert arr.std(axis=0).max() < 0.25                      # only sampling noise remains


def test_pelt_rejects_unknown_null():
    pelt = pytest.importorskip("src.chunking.methods.change_point_pelt")
    with pytest.raises(ValueError):
        pelt._run_analysis_single_attempt("x.csv", "blue", penalty_null="nope",
                                          _ikis_dict={1: np.ones(7)}, _yellow_ikis_dict={1: np.ones(7)})


def test_hsmm_duration_prior():
    pytest.importorskip("pyhsmm")
    hsmm = pytest.importorskip("src.chunking.methods.hsmm_chunking")
    data = np.random.default_rng(0).normal(0.1, 0.3, 200)
    model = hsmm._build_pyhsmm_model(4, 1, 7, 0, duration_prior_mean=2.0, duration_prior_strength=2.0,
                                     emission_prior="data", data=data)
    d = model.dur_distns[0]
    assert d.alpha_0 / d.beta_0 == pytest.approx(1.0)          # mean duration 1 + 1 = 2 IKIs
    o = model.obs_distns[0]
    assert float(np.ravel(o.mu_0)[0]) == pytest.approx(data.mean())
    legacy = hsmm._build_pyhsmm_model(4, 1, 7, 0, duration_prior_mean=None, emission_prior="fixed")
    assert legacy.dur_distns[0].alpha_0 / legacy.dur_distns[0].beta_0 == pytest.approx(7.0)
    with pytest.raises(ValueError):
        hsmm._build_pyhsmm_model(4, 1, 7, 0, duration_prior_mean=0.5)
