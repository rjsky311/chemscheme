"""ChemScheme MCP Server — Phase 8.

Wraps ChemScheme CLI tools as MCP tools for AI agent consumption.
Uses FastMCP from the MCP SDK.
Blocking work is offloaded via asyncio.to_thread (C6).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging

from mcp.server.fastmcp import FastMCP

_logger = logging.getLogger(__name__)

chemscheme_mcp = FastMCP("chemscheme")


def _sync_draw(
    scheme: dict,  # type: ignore[type-arg]
    output_format: str,
    style: str,
    dpi: int,
    theme: str | None = None,
) -> str:
    """Synchronous draw logic (runs in thread pool)."""
    from chemscheme.renderers import render_scheme
    from chemscheme.renderers.layout_engine import LayoutConfig
    from chemscheme.renderers.themes import get_theme, list_themes
    from chemscheme.schema.scheme_schema import validate_scheme

    errors = validate_scheme(scheme)
    if errors:
        return json.dumps(
            {"status": "error", "error": f"Validation failed: {'; '.join(errors)}"},
            ensure_ascii=False,
        )

    # AC-14.5: Resolve theme — explicit theme > style alias > default
    resolved_theme = None
    theme_name = "default"
    if theme is not None:
        try:
            resolved_theme = get_theme(theme)
            theme_name = theme
        except ValueError:
            available = ", ".join(list_themes())
            return json.dumps(
                {
                    "status": "error",
                    "error": f"Unknown theme '{theme}'. Available themes: {available}",
                },
                ensure_ascii=False,
            )
    elif style == "ACS2024":
        # Legacy style maps to default theme (backward compat)
        theme_name = "default"
    else:
        _logger.info("Unknown style '%s', falling back to default theme", style)
        theme_name = "default"

    config = LayoutConfig()
    result_bytes = render_scheme(
        scheme, fmt=output_format, config=config, dpi=dpi, theme=resolved_theme
    )

    if output_format == "png":
        data = base64.b64encode(result_bytes).decode("ascii")
    else:
        data = result_bytes.decode("utf-8")

    return json.dumps(
        {
            "status": "ok",
            "format": output_format,
            "theme": theme_name,
            "style": style,
            "data": data,
        },
        ensure_ascii=False,
    )


@chemscheme_mcp.tool(
    name="draw_scheme",
    description="Render a scheme.json to SVG/PNG",
)
async def draw_scheme(
    scheme: dict,  # type: ignore[type-arg]
    output_format: str = "svg",
    style: str = "ACS2024",
    dpi: int = 300,
    theme: str | None = None,
) -> str:
    """Render a scheme dict to SVG or PNG.

    Args:
        scheme: A valid scheme.json object with schema_version and steps.
        output_format: Output format, "svg" or "png".
        style: Deprecated rendering style preset. Use 'theme' instead.
        dpi: Resolution for PNG output.
        theme: Journal theme name (acs, nature, rsc, wiley, default).

    Returns:
        JSON string with status, format, theme, and data (SVG string or base64 PNG).
    """
    try:
        return await asyncio.to_thread(_sync_draw, scheme, output_format, style, dpi, theme)
    except Exception as e:
        return json.dumps(
            {"status": "error", "error": str(e)},
            ensure_ascii=False,
        )


def _sync_validate(
    scheme: dict,  # type: ignore[type-arg]
    layers: list[str] | None,
) -> str:
    """Synchronous validation logic (runs in thread pool)."""
    from chemscheme.schema.scheme_schema import validate_scheme as _validate_schema
    from chemscheme.validators.layer1_chemistry import validate_chemistry
    from chemscheme.validators.layer2_reaction import validate_reaction
    from chemscheme.validators.layer3_scheme import validate_continuity

    known_layers = {"schema", "chemistry", "reaction", "continuity"}
    requested = set(layers) if layers else {"schema", "chemistry", "reaction", "continuity"}

    unknown = requested - known_layers
    if unknown:
        return json.dumps(
            {"status": "error", "error": f"Unknown validation layers: {sorted(unknown)}"},
            ensure_ascii=False,
        )

    all_errors: list[str] = []
    all_warnings: list[str] = []
    layers_checked: list[str] = []

    if "schema" in requested:
        schema_errors = _validate_schema(scheme)
        all_errors.extend(schema_errors)
        layers_checked.append("schema")
        if schema_errors:
            return json.dumps(
                {
                    "status": "ok",
                    "valid": False,
                    "issues": all_errors,
                    "warnings": all_warnings,
                    "layers_checked": layers_checked,
                },
                ensure_ascii=False,
            )
    else:
        schema_errors = _validate_schema(scheme)
        if schema_errors:
            return json.dumps(
                {
                    "status": "ok",
                    "valid": False,
                    "issues": schema_errors,
                    "warnings": ["Schema validation failed; requested layers skipped"],
                    "layers_checked": ["schema"],
                },
                ensure_ascii=False,
            )

    steps = scheme.get("steps", [])

    if "chemistry" in requested:
        all_smiles: list[str] = []
        for step in steps:
            all_smiles.extend(step.get("reactants", []))
            all_smiles.extend(step.get("products", []))
        chem_result = validate_chemistry(all_smiles)
        all_errors.extend(chem_result.errors)
        all_warnings.extend(chem_result.warnings)
        layers_checked.append("chemistry")

    if "reaction" in requested:
        for step in steps:
            rxn_result = validate_reaction(
                reactants=step.get("reactants", []),
                products=step.get("products", []),
                reagents=step.get("arrow_above", ""),
                predictor=None,
            )
            all_errors.extend(rxn_result.errors)
            all_warnings.extend(rxn_result.warnings)
        layers_checked.append("reaction")

    if "continuity" in requested:
        cont_result = validate_continuity(scheme)
        all_errors.extend(cont_result.errors)
        all_warnings.extend(cont_result.warnings)
        layers_checked.append("continuity")

    return json.dumps(
        {
            "status": "ok",
            "valid": len(all_errors) == 0,
            "issues": all_errors,
            "warnings": all_warnings,
            "layers_checked": layers_checked,
        },
        ensure_ascii=False,
    )


@chemscheme_mcp.tool(
    name="validate_scheme",
    description="Validate a scheme.json for chemical correctness",
)
async def validate_scheme(
    scheme: dict,  # type: ignore[type-arg]
    layers: list[str] | None = None,
) -> str:
    """Validate a scheme dict for chemical correctness.

    Args:
        scheme: A scheme.json object to validate.
        layers: Validation layers to run. Options: "schema", "chemistry",
                "reaction", "continuity". Default: all four layers.

    Returns:
        JSON string with status, valid, issues, warnings, layers_checked.
    """
    try:
        return await asyncio.to_thread(_sync_validate, scheme, layers)
    except Exception as e:
        return json.dumps(
            {"status": "error", "error": str(e)},
            ensure_ascii=False,
        )


def _sync_resolve(name: str) -> str:
    """Synchronous resolve logic (runs in thread pool)."""
    from chemscheme.parsers.name_resolver import resolve_name as _resolve

    result = _resolve(name)

    if not result.smiles:  # C6 fix: base failure on empty SMILES, not confidence
        return json.dumps(
            {
                "status": "error",
                "error": f"Could not resolve '{name}'",
                "smiles": result.smiles,
                "confidence": result.confidence,
                "source": result.source,
                "warnings": result.warnings,
            },
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "status": "ok",
            "smiles": result.smiles,
            "confidence": result.confidence,
            "source": result.source,
            "warnings": result.warnings,
        },
        ensure_ascii=False,
    )


@chemscheme_mcp.tool(
    name="resolve_name",
    description="Resolve a chemical compound name to SMILES",
)
async def resolve_name(name: str) -> str:
    """Resolve a chemical compound name to SMILES.

    Args:
        name: Chemical compound name (e.g. "benzene", "aspirin").

    Returns:
        JSON string with status, smiles, confidence, source, warnings.
    """
    try:
        return await asyncio.to_thread(_sync_resolve, name)
    except Exception as e:
        return json.dumps(
            {"status": "error", "error": str(e), "warnings": []},
            ensure_ascii=False,
        )


def _sync_extract_image(image_b64: str, image_format: str) -> str:
    """Synchronous image extraction logic (runs in thread pool)."""
    import base64 as _b64
    import contextlib

    from chemscheme.recognition.mocks import MockMoleculeRecognizer
    from chemscheme.recognition.ocsr import (
        DecimerRecognizer,
        MolNexTRRecognizer,
        MolScribeRecognizer,
        recognize_molecule,
    )

    try:
        image_bytes = _b64.b64decode(image_b64, validate=True)
    except Exception as exc:
        return json.dumps(
            {"status": "error", "error": f"Invalid base64 image data: {exc}"},
            ensure_ascii=False,
        )

    if not image_bytes:
        return json.dumps(
            {"status": "error", "error": "Empty image data"},
            ensure_ascii=False,
        )

    # Build recognizer list: try real Docker/subprocess tools, mock fallback
    # R2-02 fix: verify tools actually work before adding (constructors don't check)
    from chemscheme.recognition.runner import is_tool_available

    recognizers: list = []
    _using_mock = False
    with contextlib.suppress(Exception):
        rec = MolScribeRecognizer()
        if rec._use_docker or is_tool_available(rec._cmd):  # noqa: SLF001
            recognizers.append(rec)
    with contextlib.suppress(Exception):
        dec = DecimerRecognizer()
        if dec._use_docker or is_tool_available(dec._cmd):  # noqa: SLF001
            recognizers.append(dec)
    # S68: MolNexTR (MARCUS-spec parity, Phase 17c) — highest single-tool accuracy
    with contextlib.suppress(Exception):
        mol = MolNexTRRecognizer()
        if mol._use_docker or is_tool_available(mol._cmd):  # noqa: SLF001
            recognizers.append(mol)
    if not recognizers:
        import logging as _log

        _log.getLogger(__name__).warning(
            "No real OCSR recognizers available, using mock (results are NOT real)"
        )
        recognizers.append(MockMoleculeRecognizer())
        _using_mock = True

    try:
        result = recognize_molecule(
            image=image_bytes,
            recognizers=recognizers,
            image_format=image_format,
        )
    except Exception as exc:
        return json.dumps(
            {"status": "error", "error": f"Recognition failed: {exc}"},
            ensure_ascii=False,
        )

    # A4 fix: empty SMILES = recognition failure, not success
    if not result.value:
        return json.dumps(
            {
                "status": "error",
                "error": "Recognition produced no result",
                "needs_human_review": True,
            },
            ensure_ascii=False,
        )

    resp: dict = {
        "status": "ok",
        "smiles": result.value,
        "confidence": result.confidence,
        "source": result.source,
        "agreement": result.agreement.value
        if hasattr(result.agreement, "value")
        else str(result.agreement),
    }
    if _using_mock:
        resp["warning"] = "Result from mock recognizer (not real OCSR)"
    return json.dumps(resp, ensure_ascii=False)


@chemscheme_mcp.tool(
    name="extract_image",
    description="Recognize molecule structure from an image (base64-encoded)",
)
async def extract_image(
    image_b64: str,
    image_format: str = "png",
) -> str:
    """Recognize molecule structure from a base64-encoded image.

    Args:
        image_b64: Base64-encoded image data (PNG or JPEG).
        image_format: Image format, "png" or "jpeg".

    Returns:
        JSON string with status, smiles, confidence, source, agreement.
    """
    try:
        return await asyncio.to_thread(_sync_extract_image, image_b64, image_format)
    except Exception as e:
        return json.dumps(
            {"status": "error", "error": str(e)},
            ensure_ascii=False,
        )


# ---------------------------------------------------------------------------
# Phase 19: Domain Library tools
# ---------------------------------------------------------------------------


@chemscheme_mcp.tool(
    name="lookup_reagent",
    description="Search the reagent abbreviation library by name, abbreviation, or category",
)
async def mcp_lookup_reagent(
    query: str,
    category: str | None = None,
) -> str:
    """Lookup reagents in the domain library.

    Args:
        query: Search string (case-insensitive, partial match).
        category: Optional filter (catalyst, base, solvent, reagent, ligand, etc.).

    Returns:
        JSON string with matching reagents.
    """
    try:
        import asyncio

        from chemscheme.domain_library import lookup_reagent

        results = await asyncio.to_thread(lookup_reagent, query, category=category)
        return json.dumps(
            {
                "status": "ok",
                "query": query,
                "results": [
                    {
                        "abbreviation": r.abbreviation,
                        "name": r.name,
                        "category": r.category,
                        "smiles": r.smiles,
                    }
                    for r in results
                ],
            },
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps(
            {"status": "error", "error": str(e)},
            ensure_ascii=False,
        )


@chemscheme_mcp.tool(
    name="suggest_conditions",
    description="Suggest typical conditions for a named reaction (Suzuki, Stille, Heck, etc.)",
)
async def mcp_suggest_conditions(
    reaction_type: str,
) -> str:
    """Suggest conditions for a named reaction.

    Args:
        reaction_type: Reaction name (case-insensitive, partial match).

    Returns:
        JSON string with typical conditions or available reactions list.
    """
    try:
        import asyncio

        from chemscheme.domain_library import _load_reactions, suggest_conditions

        result = await asyncio.to_thread(suggest_conditions, reaction_type)
        if result:
            return json.dumps(
                {
                    "status": "ok",
                    "name": result.name,
                    "catalyst": result.catalyst,
                    "solvents": list(result.solvents),
                    "bases": list(result.bases),
                    "reagents": list(result.reagents),
                    "temperature": result.temperature,
                    "time": result.time,
                    "atmosphere": result.atmosphere,
                    "description": result.description,
                },
                ensure_ascii=False,
            )
        all_names = [r.name for r in _load_reactions()]
        return json.dumps(
            {
                "status": "not_found",
                "query": reaction_type,
                "available_reactions": all_names,
            },
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps(
            {"status": "error", "error": str(e)},
            ensure_ascii=False,
        )


# ---------------------------------------------------------------------------
# User-extensible library (Post-Phase 20)
# ---------------------------------------------------------------------------


@chemscheme_mcp.tool()
async def add_user_reagent(
    abbreviation: str,
    name: str,
    category: str,
    smiles: str | None = None,
) -> str:
    """Add a custom reagent to the user library (~/.chemscheme/).

    Args:
        abbreviation: Short name (e.g., "MyNFA").
        name: Full name of the reagent.
        category: Category (catalyst, solvent, reagent, nfa_acceptor, etc.).
        smiles: Optional SMILES string (validated via RDKit).

    Returns:
        JSON with added reagent details or error.
    """
    import asyncio

    from chemscheme.domain_library import add_user_reagent as _add

    try:
        result = await asyncio.to_thread(_add, abbreviation, name, category, smiles=smiles)
        return json.dumps(
            {
                "status": "ok",
                "abbreviation": result.abbreviation,
                "name": result.name,
                "category": result.category,
                "smiles": result.smiles,
            },
            ensure_ascii=False,
        )
    except ValueError as e:
        return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False)


@chemscheme_mcp.tool()
async def remove_user_reagent(abbreviation: str) -> str:
    """Remove a custom reagent from the user library.

    Only custom reagents can be removed; built-in reagents are immutable.

    Args:
        abbreviation: Abbreviation of the reagent to remove.

    Returns:
        JSON with removal status.
    """
    import asyncio

    from chemscheme.domain_library import remove_user_reagent as _remove

    removed = await asyncio.to_thread(_remove, abbreviation)
    return json.dumps(
        {"status": "ok" if removed else "not_found", "abbreviation": abbreviation},
        ensure_ascii=False,
    )


@chemscheme_mcp.tool()
async def list_user_reagents() -> str:
    """List all custom reagents in the user library (~/.chemscheme/).

    Returns:
        JSON with list of user-defined reagents.
    """
    import asyncio

    from chemscheme.domain_library import list_user_reagents as _list

    reagents = await asyncio.to_thread(_list)
    return json.dumps(
        {
            "status": "ok",
            "count": len(reagents),
            "reagents": [
                {
                    "abbreviation": r.abbreviation,
                    "name": r.name,
                    "category": r.category,
                    "smiles": r.smiles,
                }
                for r in reagents
            ],
        },
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# Text extraction (Phase 20)
# ---------------------------------------------------------------------------


@chemscheme_mcp.tool()
async def extract_text(text: str, use_ai: bool = True) -> str:
    """Extract structured reaction data from experimental text.

    Parses experimental paragraphs (journal papers, SI, lab notebooks)
    into scheme.json format using regex + domain heuristics + optional AI.

    Args:
        text: Experimental text paragraph(s) to extract from.
        use_ai: Use Claude API for ambiguous role assignment (default True).

    Returns:
        JSON with {"steps": [...], "warnings": [...]}.
    """
    try:
        from chemscheme.text_extraction import extract_reactions_from_text

        result = await asyncio.to_thread(extract_reactions_from_text, text, use_ai)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps(
            {"status": "error", "error": str(e)},
            ensure_ascii=False,
        )


# ---------------------------------------------------------------------------
# Procedure generation (Phase 21)
# ---------------------------------------------------------------------------


@chemscheme_mcp.tool()
async def generate_procedure(scheme: dict) -> str:  # type: ignore[type-arg]
    """Generate experimental procedure text from a scheme.json.

    Produces publication-quality experimental section paragraphs
    from structured reaction scheme data. Uses compound_labels for naming
    and domain_library for reaction type identification.

    Args:
        scheme: A valid scheme.json object with schema_version and steps.

    Returns:
        JSON with {"text": "...", "steps": [...], "warnings": [...]}.
    """
    try:
        from chemscheme.procedure_generator import generate_procedure as _gen

        result = await asyncio.to_thread(_gen, scheme)
        return json.dumps(
            {
                "text": result.text,
                "steps": [
                    {
                        "step_number": s.step_number,
                        "text": s.text,
                        "reaction_type": s.reaction_type,
                    }
                    for s in result.steps
                ],
                "warnings": result.warnings,
            },
            ensure_ascii=False,
            indent=2,
        )
    except ValueError as e:
        return json.dumps(
            {"status": "error", "error": str(e)},
            ensure_ascii=False,
        )


if __name__ == "__main__":
    chemscheme_mcp.run()
