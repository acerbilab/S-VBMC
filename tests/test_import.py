# AI Summary: Ensures the svbmc package exposes its documented public API symbols.
import importlib


def test_public_api():
    """Package should import and expose core symbols."""
    svbmc = importlib.import_module("svbmc")

    assert hasattr(svbmc, "SVBMC"), "SVBMC class missing from public API"
    assert hasattr(svbmc, "targets"), "targets sub‑module missing from public API"
    assert hasattr(svbmc, "utils"), "utils sub‑module missing from public API"
    # Version string should be present and be a string
    assert isinstance(svbmc.__version__, str)
