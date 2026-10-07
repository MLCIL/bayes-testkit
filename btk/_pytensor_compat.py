"""Narrow compatibility fixes for PyTensor's macOS compiler configuration."""

import os
import subprocess
import sys
import warnings
from typing import Any

_COMPILER_PROBE_TIMEOUT_SECONDS = 5
_PATCH_MARKER = "_btk_unsupported_ld64_removed"
_PATCH_ENV_VAR = "BTK_PATCH_PYTENSOR"


def _pytensor_patching_enabled() -> bool:
    """Return whether the PyTensor compatibility patch is enabled."""
    return os.environ.get(_PATCH_ENV_VAR, "1") != "0"


def _compiler_links(cxx: str, linker_flags: tuple[str, ...] = ()) -> bool:
    """Return whether *cxx* can link a minimal macOS dynamic library."""
    command = [
        cxx,
        "-dynamiclib",
        "-x",
        "c++",
        "-",
        "-o",
        os.devnull,
        *linker_flags,
    ]
    try:
        result = subprocess.run(  # noqa: S603 - cxx comes from PyTensor config
            command,
            input="int btk_pytensor_link_probe;\n",
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=_COMPILER_PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def configure_pytensor_macos_linker() -> None:
    """Remove PyTensor's ``-ld64`` only when the selected compiler rejects it.

    PyTensor adds ``-ld64`` based on the reported macOS version. Newer Apple
    toolchains no longer support that linker selection flag and interpret it as
    a request for a library named ``d64``. A baseline probe prevents this shim
    from hiding unrelated compiler or SDK failures.
    """
    if sys.platform != "darwin" or not _pytensor_patching_enabled():
        return

    from pytensor import __version__ as pytensor_version
    from pytensor import config
    from pytensor.link.c.cmodule import GCC_compiler

    if not config.cxx:
        return

    compile_args = GCC_compiler.compile_args
    compile_args_function = getattr(compile_args, "__func__", compile_args)
    if getattr(compile_args_function, _PATCH_MARKER, False):
        return

    try:
        generated_flags = compile_args(march_flags=False)
    except TypeError:
        return

    should_patch = (
        "-ld64" in generated_flags
        and _compiler_links(config.cxx)
        and not _compiler_links(config.cxx, ("-ld64",))
    )
    if not should_patch:
        return

    def compatible_compile_args(
        _compiler_class: type[Any], *args: Any, **kwargs: Any
    ) -> list[str]:
        return [flag for flag in compile_args(*args, **kwargs) if flag != "-ld64"]

    setattr(compatible_compile_args, _PATCH_MARKER, True)
    GCC_compiler.compile_args = classmethod(compatible_compile_args)
    warnings.warn(
        f"Bayes Test Kit patched PyTensor {pytensor_version}'s macOS linker "
        "flags by removing unsupported -ld64 (unsupported in clang++ shipped with Xcode 27 or later). Set BTK_PATCH_PYTENSOR=0 to "
        "disable this patch.",
        UserWarning,
        stacklevel=2,
    )
