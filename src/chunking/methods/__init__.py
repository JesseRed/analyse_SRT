"""Registry of chunking methods. Add new methods here and implement run_analysis in methods/<name>.py."""

from __future__ import annotations

from typing import Callable

from .._base import ChunkingResult

# Type for run_analysis(filepath, sequence_type, **kwargs) -> ChunkingResult
ChunkingRunner = Callable[..., ChunkingResult]

REGISTRY: dict[str, ChunkingRunner] = {}
BUILTIN_METHODS = {"community_network", "change_point_pelt", "hsmm_chunking"}


def register(name: str) -> Callable[[ChunkingRunner], ChunkingRunner]:
    """Decorator to register a chunking method."""

    def decorator(fn: ChunkingRunner) -> ChunkingRunner:
        REGISTRY[name] = fn
        return fn

    return decorator


def get_runner(method_name: str) -> ChunkingRunner:
    """Return the run_analysis function for the given method."""
    if method_name not in REGISTRY:
        if method_name == "community_network":
            from . import community_network  # noqa: WPS433

            REGISTRY["community_network"] = community_network.run_analysis
        elif method_name == "change_point_pelt":
            from . import change_point_pelt  # noqa: WPS433

            REGISTRY["change_point_pelt"] = change_point_pelt.run_analysis
        elif method_name == "hsmm_chunking":
            from . import hsmm_chunking  # noqa: WPS433

            REGISTRY["hsmm_chunking"] = hsmm_chunking.run_analysis

    if method_name not in REGISTRY:
        raise ValueError(
            f"Unknown chunking method: {method_name}. "
            f"Available: {sorted(BUILTIN_METHODS)}"
        )
    return REGISTRY[method_name]


def list_methods() -> list[str]:
    """Return sorted list of registered method names."""
    return sorted(BUILTIN_METHODS.union(REGISTRY))

__all__ = ["REGISTRY", "register", "get_runner", "list_methods", "ChunkingRunner"]
