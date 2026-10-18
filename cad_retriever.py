"""Compatibility shim: the CAD retrieval engine now lives in cad_harvester.py (with author attribution)."""

from cad_harvester import (  # noqa: F401
    credits,
    delete_cached,
    candidates,
    generate_from_specs,
    harvest,
    retrieve,
    select,
    upload_glb,
    validate_glb,
    write_credits_md,
)
