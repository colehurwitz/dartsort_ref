"""Tests for issue #13 (adaptive sub-batching) and #14 (Decollider pickle)."""

import pickle

import numpy as np
import torch

import dartsort
from dartsort.config import DARTsortUserConfig
from dartsort.transform._multichan_denoiser_kit import BaseMultichannelDenoiser
from dartsort.transform.decollider import Decollider
from dartsort.util.internal_config import MatchingConfig, default_waveform_cfg
from dartsort.util.waveform_util import make_channel_index


def _make_channel_index_and_geom(n_channels=10, radius=100.0):
    geom = np.c_[np.zeros(n_channels), np.arange(n_channels) * 25.0]
    geom_t = torch.from_numpy(geom).float()
    ci = make_channel_index(geom, radius)
    ci_t = torch.from_numpy(ci)
    return ci_t, geom_t


# -- Issue #14: Pickle serialization tests --


def test_base_multichannel_denoiser_pickle_cpu():
    """BaseMultichannelDenoiser round-trips through pickle on CPU."""
    ci, geom = _make_channel_index_and_geom()
    net = BaseMultichannelDenoiser(
        channel_index=ci,
        geom=geom,
        waveform_cfg=default_waveform_cfg,
    )
    assert str(net.device) == "cpu"

    data = pickle.dumps(net)
    net2 = pickle.loads(data)

    assert str(net2.device) == "cpu"
    assert torch.equal(net2.b.channel_index, net.b.channel_index)


def test_decollider_pickle_cpu():
    """Decollider round-trips through pickle on CPU."""
    ci, geom = _make_channel_index_and_geom()
    dec = Decollider(
        channel_index=ci,
        geom=geom,
        waveform_cfg=default_waveform_cfg,
    )
    assert str(dec.device) == "cpu"

    data = pickle.dumps(dec)
    dec2 = pickle.loads(data)

    assert str(dec2.device) == "cpu"
    assert torch.equal(dec2.b.channel_index, dec.b.channel_index)
    assert dec2.step_callback is None


def test_decollider_pickle_preserves_state():
    """Decollider preserves key attributes through pickle."""
    ci, geom = _make_channel_index_and_geom()
    dec = Decollider(
        channel_index=ci,
        geom=geom,
        waveform_cfg=default_waveform_cfg,
        hidden_dims=(256, 256),
        inference_kind="amortized",
    )

    data = pickle.dumps(dec)
    dec2 = pickle.loads(data)

    assert dec2.hidden_dims == (256, 256)
    assert dec2.inference_kind == "amortized"
    assert dec2._needs_fit == dec._needs_fit


@torch.no_grad()
def test_decollider_pickle_cuda():
    """Decollider round-trips through pickle on CUDA, preserving device."""
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("CUDA not available")

    ci, geom = _make_channel_index_and_geom()
    dec = Decollider(
        channel_index=ci,
        geom=geom,
        waveform_cfg=default_waveform_cfg,
    )
    dec = dec.to("cuda:0")
    assert dec.device == torch.device("cuda", 0)

    data = pickle.dumps(dec)
    dec2 = pickle.loads(data)

    assert dec2.device == torch.device("cuda", 0)
    assert dec2.b.channel_index.device.type == "cuda"


# -- Issue #13: Config and sub-batching tests --


def test_max_spikes_per_batch_default():
    """MatchingConfig has max_spikes_per_batch with correct default."""
    cfg = MatchingConfig()
    assert cfg.max_spikes_per_batch == 16384


def test_max_spikes_per_batch_no_old_name():
    """Old max_spikes_per_second field no longer exists."""
    cfg = MatchingConfig()
    assert not hasattr(cfg, "max_spikes_per_second")


def test_user_config_max_spikes_per_batch():
    """DARTsortUserConfig exposes max_spikes_per_batch."""
    cfg = DARTsortUserConfig()
    assert cfg.max_spikes_per_batch == 16384


def test_user_config_custom_max_spikes():
    """DARTsortUserConfig accepts custom max_spikes_per_batch."""
    cfg = DARTsortUserConfig(max_spikes_per_batch=8192)
    assert cfg.max_spikes_per_batch == 8192


def test_config_propagation():
    """max_spikes_per_batch propagates from user config to internal config."""
    user_cfg = DARTsortUserConfig(max_spikes_per_batch=8192)
    internal_cfg = dartsort.to_internal_config(user_cfg, 10)
    assert internal_cfg.matching_cfg.max_spikes_per_batch == 8192


def test_config_default_propagation():
    """Default max_spikes_per_batch propagates correctly."""
    user_cfg = DARTsortUserConfig()
    internal_cfg = dartsort.to_internal_config(user_cfg, 10)
    assert internal_cfg.matching_cfg.max_spikes_per_batch == 16384
