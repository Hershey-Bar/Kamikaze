"""Compatibility shim: the CAD retrieval engine now lives in cad_harvester.py (with author attribution)."""

from cad_harvester import (  # noqa: F401
    credits,
    delete_cached,
    ensure_base_models,
    harvest,
    retrieve,
    write_credits_md,
)
