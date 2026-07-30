import contextlib
import dataclasses
import os
import shutil
import signal
import subprocess
import threading
from importlib.resources.abc import Traversable
from pathlib import Path
from time import perf_counter
from typing import dataclass_transform, NoReturn

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass as pydantic_dataclass

from .logging_util import DARTSORTVERBOSE, get_logger

logger = get_logger(__name__)


# without this, pydantic allows unknown keys, so you can easily
# make a typo in your parameter name!
pydantic_strict_cfg = ConfigDict(strict=True, extra="forbid")


# used for configuration objects in [internal_]config.py
@dataclass_transform(kw_only_default=True, frozen_default=True)
def cfg_dataclass(*args, frozen=True, kw_only=True, **kwargs):
    return pydantic_dataclass(
        *args, **kwargs, frozen=frozen, kw_only=kw_only, config=pydantic_strict_cfg
    )


# lightweight dataclass defaults
@dataclass_transform(kw_only_default=True, eq_default=False)
def databag(*args, slots=True, kw_only=True, eq=False, repr=False, **kwargs):
    return dataclasses.dataclass(
        *args, **kwargs, slots=slots, kw_only=kw_only, eq=eq, repr=repr
    )


# random utility classes

_timer_stack = []

# Module-level profiling flags. When enable_profiling is True, both NVTX and
# CUDA event timing are activated (if CUDA is available). These can also be
# controlled individually for advanced use.
enable_profiling: bool = False
enable_nvtx: bool = False
enable_cuda_timing: bool = False

_cuda_available: bool | None = None


def _check_cuda() -> bool:
    global _cuda_available
    if _cuda_available is None:
        try:
            import torch
            _cuda_available = torch.cuda.is_available()
        except Exception:
            _cuda_available = False
    return _cuda_available


class timer:
    """
    with timer("hi"):
        bubblesort(np.arange(1e6)[::-1])
    # prints: hi took rot90(8) s
    with timer("zoom", {}) as tic:
        with timer("zip") as tac:
            pass
    assert np.isclose(tac.dt, 0)
    tic.results_dict # => nested timings

    When enable_profiling or enable_nvtx is True and CUDA is available,
    each timer context also pushes/pops an NVTX range (visible in Nsight
    Systems). When enable_profiling or enable_cuda_timing is True, GPU
    wall-clock time is recorded via cuda.Events and stored in
    results_dict with a "_cuda" suffix.
    """

    def __init__(self, name="timer", results_dict=None, loglevel=DARTSORTVERBOSE):
        self.loglevel = loglevel
        self.name = name
        self.results_dict = results_dict
        self.parent = None
        self._nvtx_active = False
        self._cuda_events_active = False
        self._start_event = None
        self._end_event = None

    def start(self):
        self.t0 = perf_counter()
        if (enable_profiling or enable_nvtx) and _check_cuda():
            import torch.cuda
            torch.cuda.nvtx.range_push(self.name)
            self._nvtx_active = True
        if (enable_profiling or enable_cuda_timing) and _check_cuda():
            import torch.cuda
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
            self._cuda_events_active = True

    def stop(self):
        if self._nvtx_active:
            import torch.cuda
            torch.cuda.nvtx.range_pop()
            self._nvtx_active = False
        if self._cuda_events_active:
            self._end_event.record()
        self.dt = perf_counter() - self.t0
        logger.log(self.loglevel, "%s took %ss", self.name, self.dt)
        if self.parent is not None and self.results_dict is not None:
            self.results_dict[f"{self.parent.name}: {self.name}"] = self.dt
        elif self.results_dict is not None:
            self.results_dict[self.name] = self.dt
        if self._cuda_events_active:
            import torch.cuda
            torch.cuda.synchronize()
            cuda_dt = self._start_event.elapsed_time(self._end_event) / 1000.0
            key = f"{self.name}_cuda"
            if self.parent is not None and self.results_dict is not None:
                self.results_dict[f"{self.parent.name}: {key}"] = cuda_dt
            elif self.results_dict is not None:
                self.results_dict[key] = cuda_dt
            self._cuda_events_active = False

    def __enter__(self):
        global _timer_stack
        if len(_timer_stack):
            self.parent = _timer_stack[-1]
            if self.results_dict is None:
                self.results_dict = self.parent.results_dict
        _timer_stack.append(self)
        self.start()
        return self

    def __exit__(self, *args):
        global _timer_stack
        self.stop()
        assert _timer_stack.pop() is self
        self.parent = None


def nvtx_range(name: str):
    """Lightweight NVTX-only context manager for hot-path annotations.

    Unlike timer, this does NOT record CPU/CUDA times — it only emits
    NVTX ranges visible in Nsight Systems. Zero cost when profiling is off.
    """
    if (enable_profiling or enable_nvtx) and _check_cuda():
        return _NvtxRange(name)
    return contextlib.nullcontext()


class _NvtxRange:
    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        import torch.cuda
        torch.cuda.nvtx.range_push(self.name)
        return self

    def __exit__(self, *args):
        import torch.cuda
        torch.cuda.nvtx.range_pop()


class NoKeyboardInterrupt:
    """A context manager that we use to avoid ending up in invalid states."""

    def handler(self, *sig):
        if self.sig:
            signal.signal(signal.SIGINT, self.old_handler)
            sig, self.sig = self.sig, None
            self.old_handler(*sig)  # type: ignore
        self.sig = sig

    def __enter__(self):
        # TODO: maybe should just handle this in the peeling code. would need to detect
        # partially saved batches somehow... uhh... maybe based on the last sample saved,
        # if we update that last and resume from there...
        if threading.current_thread() is threading.main_thread() and os.name == "posix":
            self.old_handler = signal.signal(signal.SIGINT, self.handler)
            self.sig = None

    def __exit__(self, type, value, traceback):
        if threading.current_thread() is threading.main_thread() and os.name == "posix":
            signal.signal(signal.SIGINT, self.old_handler)
            if self.sig:
                self.old_handler(*self.sig)  # type: ignore


if threading.current_thread() is threading.main_thread() and os.name == "posix":
    delay_keyboard_interrupt = NoKeyboardInterrupt()
else:
    delay_keyboard_interrupt = contextlib.nullcontext()


# files and paths


def ensure_path(
    p: str | Path | Traversable | None,
    strict=False,
    mkdir=False,
    parents=False,
    resolve=False,
) -> Path:
    if p is None:
        raise ValueError("Can't resolve path None.")
    if isinstance(p, Traversable):
        assert isinstance(p, Path)
    p = Path(p)
    p = p.expanduser()
    p = p.absolute()
    if resolve:
        p = p.resolve(strict=strict)
    elif strict:
        assert p.exists()
    if mkdir:
        p.mkdir(parents=parents, exist_ok=True)
    return p


def dartcopy2(icfg, src, dest):
    if icfg.workdir_copier == "shutil":
        try:
            shutil.copy2(src, dest, follow_symlinks=icfg.workdir_follow_symlinks)
        except shutil.SameFileError:
            # this happens in a symlink workflow that I use sometimes
            return
    elif icfg.workdir_copier == "rsync":
        _rsync(src, dest, archive=False, follow_symlinks=icfg.workdir_follow_symlinks)
    else:
        assert False


def dartcopytree(icfg, src, dest):
    try:
        if icfg.workdir_copier == "shutil":
            shutil.copytree(
                src,
                dest,
                symlinks=not icfg.workdir_follow_symlinks,
                dirs_exist_ok=True,
            )
        elif icfg.workdir_copier == "rsync":
            _rsync(
                f"{src}/",
                f"{dest}/",
                archive=True,
                follow_symlinks=icfg.workdir_follow_symlinks,
            )
        else:
            assert False
    except shutil.SameFileError:
        logger.dartsortdebug(
            f"Skip dartcopytree {src} -> {dest} since shutil says they're the same."
        )
    except shutil.Error as e:
        # Sometimes the same file error is hiding in this.
        # This is probably not the right way to do this?
        # It's a weird exception format though...
        arg = e.args[0]
        if not all(isinstance(a, str) and len(a) == 1 for a in arg):
            raise
        arg = "".join(arg)
        if not arg.startswith("<DirEntry"):
            raise
        if not arg.endswith("are the same file"):
            raise
        logger.dartsortdebug(
            f"shutil.copytree said (re: {src} and {dest}) that {arg}. "
            "Ignoring that and continuing."
        )
    except Exception as e:
        raise ValueError(
            f"dartcopytree {src} -> {dest} failed. {src.exists()=}, {dest.exists()=}."
        ) from e


def _rsync(src, dest, archive=True, follow_symlinks=False, excludes=None, vp=False):
    archive_flags = ["-a" + ("vP" if vp else "")] if archive else []
    link_flags = ["--no-links", "-L"] if follow_symlinks else []
    exclude_flags = [f"--exclude={ex}" for ex in (excludes or [])]
    cmd = ["rsync", *archive_flags, *link_flags, *exclude_flags, str(src), str(dest)]
    if vp:
        logger.info(" ".join(cmd))
    res = subprocess.run(cmd)
    assert not res.returncode


def panic(msg="") -> NoReturn:
    raise AssertionError(msg)
