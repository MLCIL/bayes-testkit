"""Tests for the PyTensor macOS linker compatibility shim."""

import pytest
from pytensor import config
from pytensor.link.c import cmodule

from btk import _pytensor_compat as compat


def _compiler_with_flags(flags):
    """Create an isolated stand-in for the private PyTensor compiler class."""

    class TestCompiler:
        @classmethod
        def compile_args(cls, march_flags: bool = True) -> list[str]:
            del cls, march_flags
            return list(flags)

    return TestCompiler


def _configure_test_compiler(monkeypatch, compiler, probe):
    monkeypatch.setattr(compat.sys, "platform", "darwin")
    monkeypatch.setattr(cmodule, "GCC_compiler", compiler)
    monkeypatch.setattr(config, "cxx", "test-clang++")
    monkeypatch.setattr(compat, "_compiler_links", probe)
    monkeypatch.setenv("BTK_PATCH_PYTENSOR", "1")


def test_non_macos_platform_does_not_change_compiler(monkeypatch):
    compiler = _compiler_with_flags(["-O3", "-ld64", "-fPIC"])
    monkeypatch.setattr(cmodule, "GCC_compiler", compiler)
    monkeypatch.setattr(compat.sys, "platform", "linux")
    monkeypatch.setattr(
        compat,
        "_compiler_links",
        lambda *_args, **_kwargs: pytest.fail("compiler probe should not run"),
    )

    compat.configure_pytensor_macos_linker()

    assert compiler.compile_args() == ["-O3", "-ld64", "-fPIC"]


def test_environment_variable_disables_linker_patch(monkeypatch):
    compiler = _compiler_with_flags(["-O3", "-ld64", "-fPIC"])
    _configure_test_compiler(
        monkeypatch,
        compiler,
        lambda *_args, **_kwargs: pytest.fail("compiler probe should not run"),
    )
    monkeypatch.setenv("BTK_PATCH_PYTENSOR", "0")

    compat.configure_pytensor_macos_linker()

    assert compiler.compile_args() == ["-O3", "-ld64", "-fPIC"]


def test_compiler_without_ld64_is_not_probed(monkeypatch):
    compiler = _compiler_with_flags(["-O3", "-fPIC"])
    _configure_test_compiler(
        monkeypatch,
        compiler,
        lambda *_args, **_kwargs: pytest.fail("compiler probe should not run"),
    )

    compat.configure_pytensor_macos_linker()

    assert compiler.compile_args() == ["-O3", "-fPIC"]


def test_ld64_is_preserved_when_compiler_accepts_it(monkeypatch):
    compiler = _compiler_with_flags(["-O3", "-ld64", "-fPIC"])
    _configure_test_compiler(monkeypatch, compiler, lambda _cxx, _flags=(): True)

    compat.configure_pytensor_macos_linker()

    assert "-ld64" in compiler.compile_args()


def test_ld64_is_preserved_when_baseline_linking_fails(monkeypatch):
    compiler = _compiler_with_flags(["-O3", "-ld64", "-fPIC"])
    _configure_test_compiler(monkeypatch, compiler, lambda _cxx, _flags=(): False)

    compat.configure_pytensor_macos_linker()

    assert "-ld64" in compiler.compile_args()


def test_ld64_is_removed_when_it_is_the_only_link_failure(monkeypatch):
    calls = []
    compiler = _compiler_with_flags(["-O3", "-ld64", "-fPIC"])

    def probe(_cxx, flags=()):
        calls.append(flags)
        return flags != ("-ld64",)

    _configure_test_compiler(monkeypatch, compiler, probe)

    with pytest.warns(UserWarning, match="patched PyTensor.*BTK_PATCH_PYTENSOR=0"):
        compat.configure_pytensor_macos_linker()

    assert calls == [(), ("-ld64",)]
    assert compiler.compile_args() == ["-O3", "-fPIC"]


def test_linker_patch_is_applied_only_once(monkeypatch):
    calls = []
    compiler = _compiler_with_flags(["-O3", "-ld64", "-fPIC"])

    def probe(_cxx, flags=()):
        calls.append(flags)
        return flags != ("-ld64",)

    _configure_test_compiler(monkeypatch, compiler, probe)

    with pytest.warns(UserWarning, match="patched PyTensor"):
        compat.configure_pytensor_macos_linker()
    compat.configure_pytensor_macos_linker()

    assert calls == [(), ("-ld64",)]
    assert compiler.compile_args() == ["-O3", "-fPIC"]
