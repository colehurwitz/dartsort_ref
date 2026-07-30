"""Tests for the profiling extensions in py_util (NVTX + CUDA event timing)."""

from unittest import mock

import pytest

from dartsort.util import py_util


@pytest.fixture(autouse=True)
def _reset_profiling_flags():
    """Reset module-level profiling flags before and after each test."""
    py_util.enable_profiling = False
    py_util.enable_nvtx = False
    py_util.enable_cuda_timing = False
    old_cuda = py_util._cuda_available
    yield
    py_util.enable_profiling = False
    py_util.enable_nvtx = False
    py_util.enable_cuda_timing = False
    py_util._cuda_available = old_cuda


class TestTimerBasic:
    """Basic timer behaviour (no profiling)."""

    def test_timer_records_dt(self):
        d = {}
        with py_util.timer("basic", d) as t:
            pass
        assert t.dt >= 0
        assert "basic" in d

    def test_timer_nesting(self):
        d = {}
        with py_util.timer("outer", d):
            with py_util.timer("inner", d):
                pass
        assert "outer" in d
        assert "outer: inner" in d


class TestTimerNvtx:
    """Timer NVTX range push/pop when enable_nvtx is True."""

    def test_nvtx_push_pop_called(self):
        py_util._cuda_available = True
        py_util.enable_nvtx = True

        with mock.patch("torch.cuda.nvtx.range_push") as mock_push, \
             mock.patch("torch.cuda.nvtx.range_pop") as mock_pop:
            with py_util.timer("test_nvtx"):
                mock_push.assert_called_once_with("test_nvtx")
            mock_pop.assert_called_once()

    def test_nvtx_not_called_when_disabled(self):
        py_util._cuda_available = True
        py_util.enable_nvtx = False
        py_util.enable_profiling = False

        d = {}
        with py_util.timer("no_nvtx", d) as t:
            assert not t._nvtx_active


class TestTimerCudaEvents:
    """Timer CUDA event timing when enable_cuda_timing is True."""

    def test_cuda_events_recorded(self):
        py_util._cuda_available = True
        py_util.enable_cuda_timing = True

        mock_event_start = mock.MagicMock()
        mock_event_end = mock.MagicMock()
        mock_event_start.elapsed_time.return_value = 42.0
        events = iter([mock_event_start, mock_event_end])

        with mock.patch("torch.cuda.Event", side_effect=lambda **kw: next(events)), \
             mock.patch("torch.cuda.nvtx.range_push"), \
             mock.patch("torch.cuda.nvtx.range_pop"), \
             mock.patch("torch.cuda.synchronize"):
            d = {}
            with py_util.timer("gpu_op", d):
                pass

        assert "gpu_op_cuda" in d
        assert d["gpu_op_cuda"] == pytest.approx(0.042)
        mock_event_start.record.assert_called_once()
        mock_event_end.record.assert_called_once()


class TestNvtxRange:
    """Tests for the lightweight nvtx_range() context manager."""

    def test_nvtx_range_noop_when_disabled(self):
        py_util.enable_profiling = False
        py_util.enable_nvtx = False
        ctx = py_util.nvtx_range("noop")
        with ctx:
            pass

    def test_nvtx_range_pushes_when_enabled(self):
        py_util._cuda_available = True
        py_util.enable_nvtx = True

        with mock.patch("torch.cuda.nvtx.range_push") as mock_push, \
             mock.patch("torch.cuda.nvtx.range_pop") as mock_pop:
            ctx = py_util.nvtx_range("hot_path")
            with ctx:
                mock_push.assert_called_once_with("hot_path")
            mock_pop.assert_called_once()

    def test_nvtx_range_no_cuda(self):
        py_util._cuda_available = False
        py_util.enable_nvtx = True
        ctx = py_util.nvtx_range("no_gpu")
        with ctx:
            pass


class TestCheckCuda:
    """Tests for the _check_cuda helper."""

    def test_caches_result(self):
        py_util._cuda_available = None
        with mock.patch("torch.cuda.is_available", return_value=False):
            result = py_util._check_cuda()
            assert result is False
            assert py_util._cuda_available is False
