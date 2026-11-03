"""ChemScheme CLI entry point."""

import json
import sys
import tempfile
from pathlib import Path

import click

from chemscheme import __version__


@click.group()
@click.version_option(version=__version__, prog_name="chemscheme")
def main() -> None:
    """Generate publication-quality multi-step reaction schemes."""


@main.command()
@click.argument("scheme_file", type=click.Path(exists=True))
@click.option("-o", "--output", required=True, type=click.Path(), help="Output file (.svg/.png)")
@click.option("--json", "json_output", is_flag=True, help="Output JSON result instead of text")
@click.option("--force", is_flag=True, help="Overwrite existing output file")
@click.option(
    "--validate", is_flag=True, help="Run full chemistry/continuity validation before rendering"
)
@click.option("--theme", "theme_name", default=None, help="Journal theme (acs, nature, rsc, wiley)")
@click.option(
    "--max-row-width",
    "max_row_width",
    type=float,
    default=None,
    help=(
        "G2 column-wrap (S68): wrap multi-step schemes when row width exceeds this "
        "budget (px). Overrides 'layout: horizontal' in scheme.json. Typical: 600 "
        "(single-col journal), 1200 (double-col)."
    ),
)
@click.option(
    "--atoms-as-text",
    "atoms_as_text",
    is_flag=True,
    default=False,
    help=(
        "G4 (S69): render heteroatom labels as editable SVG <text> (custom "
        "drawer) instead of RDKit path glyphs. Allows vector editors "
        "(Illustrator, Inkscape) to treat atom symbols as text. Carbons "
        "remain implicit per chemistry convention."
    ),
)
def draw(
    scheme_file: str,
    output: str,
    json_output: bool,
    force: bool,
    validate: bool,
    theme_name: str | None,
    max_row_width: float | None,
    atoms_as_text: bool,
) -> None:
    """Render a scheme.json file to SVG or PNG."""
    from chemscheme.renderers import render_scheme
    from chemscheme.renderers.layout_engine import LayoutConfig
    from chemscheme.renderers.themes import get_theme, list_themes
    from chemscheme.schema.scheme_schema import load_scheme, validate_scheme

    # Resolve theme
    theme = None
    if theme_name is not None:
        try:
            theme = get_theme(theme_name)
        except ValueError:
            available = ", ".join(list_themes())
            msg = f"Unknown theme '{theme_name}'. Available themes: {available}"
            if json_output:
                click.echo(json.dumps({"status": "error", "error": msg}, ensure_ascii=False))
            else:
                click.echo(f"Error: {msg}", err=True)
            sys.exit(1)

    # G4 (S69): override theme to enable text-based atom rendering
    if atoms_as_text:
        from dataclasses import replace

        from chemscheme.renderers.themes import DEFAULT_THEME

        base_theme = theme if theme is not None else DEFAULT_THEME
        theme = replace(base_theme, molecule_atoms_as_text=True)

    out_path = Path(output)
    scheme_path = Path(scheme_file)

    # MR12-G2: Reject directory as output path
    if out_path.is_dir():
        msg = f"Output path '{out_path}' is a directory."
        if json_output:
            click.echo(json.dumps({"status": "error", "error": msg}, ensure_ascii=False))
        else:
            click.echo(f"Error: {msg}", err=True)
        sys.exit(1)

    # Overwrite protection
    if out_path.exists() and not force:
        msg = f"Output file already exists: {out_path}. Use --force to overwrite."
        if json_output:
            click.echo(json.dumps({"status": "error", "error": msg}, ensure_ascii=False))
        else:
            click.echo(f"Error: {msg}", err=True)
        sys.exit(1)

    # Load and validate scheme
    try:
        scheme = load_scheme(scheme_path)
    except ValueError as e:  # MR5-C3: catch malformed JSON
        msg = str(e)
        if json_output:
            click.echo(json.dumps({"status": "error", "error": msg}, ensure_ascii=False))
        else:
            click.echo(f"Error: {msg}", err=True)
        sys.exit(1)
    errors = validate_scheme(scheme)
    if errors:
        msg = f"Validation failed: {'; '.join(errors)}"
        if json_output:
            data = {"status": "error", "error": msg, "issues": errors}
            click.echo(json.dumps(data, ensure_ascii=False))
        else:
            click.echo(f"Error: {msg}", err=True)
        sys.exit(1)

    # AC-13.5: Optional full validation gate before rendering
    if validate:
        from chemscheme.schema.scheme_schema import validate_scheme_json

        full_result = validate_scheme_json(scheme)
        if not full_result["valid"]:
            issues = full_result.get("issues", [])
            msg = f"Validation failed: {'; '.join(issues)}"
            if json_output:
                click.echo(
                    json.dumps(
                        {"status": "error", "error": msg, "validation": full_result},
                        ensure_ascii=False,
                    )
                )
            else:
                click.echo(f"Error: {msg}", err=True)
            sys.exit(1)

    # Determine format from extension (MR4-2; G13 S68 added .pdf + .eps)
    suffix = out_path.suffix.lower()
    if suffix == ".png":
        fmt = "png"
    elif suffix == ".svg":
        fmt = "svg"
    elif suffix == ".pdf":
        fmt = "pdf"
    elif suffix == ".eps":
        fmt = "eps"
    else:
        msg = f"Unsupported output format '{suffix}'. Use .svg, .png, .pdf, or .eps."
        if json_output:
            click.echo(json.dumps({"status": "error", "error": msg}, ensure_ascii=False))
        else:
            click.echo(f"Error: {msg}", err=True)
        sys.exit(1)

    # Auto-create output directory (AC-4.9)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # G2 (S68): build layout config when --max-row-width provided. Setting
    # max_steps_per_row=999 lets the width budget be the sole break trigger,
    # and overrides scheme.json `layout: horizontal` (which would otherwise
    # force single-row in the engine's default config path).
    layout_config: LayoutConfig | None = None
    if max_row_width is not None:
        layout_config = LayoutConfig(
            max_row_width=max_row_width,
            max_steps_per_row=999,
        )

    # Atomic write: write to temp file then replace (AC-4.10, Windows-safe)
    try:
        result_bytes = render_scheme(scheme, fmt=fmt, theme=theme, config=layout_config)
    except (ValueError, RuntimeError) as e:
        msg = f"Rendering failed: {e}"
        if json_output:
            click.echo(json.dumps({"status": "error", "error": msg}, ensure_ascii=False))
        else:
            click.echo(f"Error: {msg}", err=True)
        sys.exit(1)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(out_path.parent), suffix=out_path.suffix)
    try:
        with open(tmp_fd, "wb") as f:
            f.write(result_bytes)
        # Retry replace on Windows PermissionError (MR4.1-P1b)
        import time

        for attempt in range(3):
            try:
                Path(tmp_path).replace(out_path)
                break
            except PermissionError:
                if attempt == 2:
                    raise
                time.sleep(0.1)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise

    if json_output:
        click.echo(
            json.dumps(
                {
                    "status": "ok",
                    "output_path": str(out_path),
                    "format": fmt,
                    "size_bytes": out_path.stat().st_size,
                },
                ensure_ascii=False,
            )
        )
    else:
        click.echo(f"Wrote {fmt.upper()} to {out_path} ({out_path.stat().st_size} bytes)")


@main.command()
def themes() -> None:
    """List available journal themes."""
    from chemscheme.renderers.themes import get_theme, list_themes

    for name in list_themes():
        theme = get_theme(name)
        click.echo(f"  {name:10s}  {theme.description}")


@main.command()
@click.argument("scheme_file", type=click.Path(exists=True))
@click.option("--json", "json_output", is_flag=True, help="Output JSON result instead of text")
def validate(scheme_file: str, json_output: bool) -> None:
    """Validate a scheme.json file."""
    from chemscheme.schema.scheme_schema import load_scheme, validate_scheme_json

    scheme_path = Path(scheme_file)
    try:
        scheme = load_scheme(scheme_path)
    except ValueError as e:  # MR5-C3: catch malformed JSON
        if json_output:
            click.echo(json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False))
        else:
            click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # Always run full validation (schema + chemistry + continuity) (MR3-3)
    result = validate_scheme_json(scheme)

    if json_output:
        click.echo(json.dumps(result, ensure_ascii=False))
        if not result["valid"]:
            sys.exit(1)
    else:
        if not result["valid"]:
            click.echo("Validation FAILED:", err=True)
            for err in result.get("issues", []):
                click.echo(f"  - {err}", err=True)
            for w in result.get("warnings", []):
                click.echo(f"  ⚠ {w}", err=True)
            sys.exit(1)
        else:
            warnings = result.get("warnings", [])
            click.echo(f"Valid scheme. (layers: {', '.join(result['layers_checked'])})")
            for w in warnings:
                click.echo(f"  ⚠ {w}", err=True)


@main.command()
@click.argument("name")
@click.option("--json", "json_output", is_flag=True, help="Output JSON result instead of text")
def resolve(name: str, json_output: bool) -> None:
    """Resolve a compound name to SMILES."""
    from chemscheme.parsers.name_resolver import resolve_name

    result = resolve_name(name)

    if json_output:
        import json as json_mod

        data = {
            "status": "ok" if result.smiles else "error",  # MR5R2-C7: consistent status field
            "smiles": result.smiles,
            "confidence": result.confidence,
            "source": result.source,
            "warnings": result.warnings,
        }
        click.echo(json_mod.dumps(data, ensure_ascii=False))
    else:
        if result.smiles:
            msg = f"{result.smiles}  (source: {result.source}, confidence: {result.confidence})"
            click.echo(msg)
        else:
            click.echo(f"Error: {'; '.join(result.warnings)}", err=True)

    if not result.smiles:  # MR5-C6: exit on empty SMILES, not confidence
        sys.exit(1)


@main.command()
@click.argument("description")
@click.option("-o", "--output", "output_file", type=click.Path(), help="Save scheme.json to file")
@click.option("--force", is_flag=True, help="Overwrite existing output file")
def ai(description: str, output_file: str | None, force: bool) -> None:
    """Generate scheme.json from a natural language description."""
    from chemscheme.parsers.ai_intent import AiServiceError, parse_intent

    # Overwrite guard (MR4.1-P1f)
    if output_file:
        out_path = Path(output_file)
        if out_path.exists() and not force:
            msg = f"Output file already exists: {out_path}. Use --force to overwrite."
            click.echo(json.dumps({"status": "error", "error": msg}, ensure_ascii=False))
            sys.exit(1)

    try:
        scheme = parse_intent(description)
    except AiServiceError as e:
        click.echo(json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False))
        sys.exit(1)

    scheme_json = json.dumps(scheme, indent=2, ensure_ascii=False)

    if output_file:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(scheme_json, encoding="utf-8")
        click.echo(f"Scheme saved to {out_path}")
    else:
        click.echo(scheme_json)


@main.command()
@click.argument("scheme_file", type=click.Path(exists=True))
@click.option("--add-step", "add_step_json", type=str, help="JSON string for new step to add")
@click.option("--remove-step", "remove_step_id", type=str, help="Step ID to remove")
@click.option(
    "--position", type=int, default=None, help="Insert position for --add-step (0-indexed)"
)
@click.option("--json", "json_output", is_flag=True, help="Output JSON result instead of text")
def edit(
    scheme_file: str,
    add_step_json: str | None,
    remove_step_id: str | None,
    position: int | None,
    json_output: bool,
) -> None:
    """Edit a scheme.json: add or remove steps."""
    from chemscheme.editors.scheme_editor import add_step as do_add
    from chemscheme.editors.scheme_editor import remove_step as do_remove
    from chemscheme.schema.scheme_schema import load_scheme, save_scheme, validate_scheme

    scheme_path = Path(scheme_file)
    try:
        scheme = load_scheme(scheme_path)
    except ValueError as e:  # MR5-C3: catch malformed JSON
        if json_output:
            click.echo(json.dumps({"status": "error", "error": str(e)}))
        else:
            click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # Validate scheme before editing (MR5-C4)
    schema_errors = validate_scheme(scheme)
    if schema_errors:
        msg = f"Scheme validation failed: {'; '.join(schema_errors)}"
        if json_output:
            click.echo(json.dumps({"status": "error", "error": msg, "issues": schema_errors}))
        else:
            click.echo(f"Error: {msg}", err=True)
        sys.exit(1)

    if add_step_json and remove_step_id:
        msg = "Cannot use --add-step and --remove-step together"
        if json_output:
            click.echo(json.dumps({"status": "error", "error": msg}))
        else:
            click.echo(f"Error: {msg}", err=True)
        sys.exit(1)

    if add_step_json:
        try:
            new_step = json.loads(add_step_json)
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON for --add-step: {e}"
            if json_output:
                click.echo(json.dumps({"status": "error", "error": msg}))
            else:
                click.echo(f"Error: {msg}", err=True)
            sys.exit(1)

        try:
            result, validation = do_add(scheme, new_step, position=position, validate=True)
        except ValueError as e:
            msg = str(e)
            if json_output:
                click.echo(json.dumps({"status": "error", "error": msg}))
            else:
                click.echo(f"Error: {msg}", err=True)
            sys.exit(1)

        # Block save on validation failure (MR5R2-C1)
        if not validation["valid"]:
            msg = f"Edit produced invalid scheme: {'; '.join(validation.get('issues', []))}"
            if json_output:
                click.echo(json.dumps({"status": "error", "error": msg, "validation": validation}))
            else:
                click.echo(f"Error: {msg}", err=True)
            sys.exit(1)

        save_scheme(result, scheme_path)

        if json_output:
            click.echo(
                json.dumps(
                    {"status": "ok", "action": "add_step", "validation": validation},
                    ensure_ascii=False,
                )
            )
        else:
            click.echo(f"Added step '{new_step.get('id', '?')}' to {scheme_path}")
            if not validation["valid"]:
                click.echo("Warning: validation issues found after edit:", err=True)
                for issue in validation.get("issues", []):
                    click.echo(f"  - {issue}", err=True)

    elif remove_step_id:
        try:
            result, validation = do_remove(scheme, remove_step_id, validate=True)
        except ValueError as e:
            msg = str(e)
            if json_output:
                click.echo(json.dumps({"status": "error", "error": msg}))
            else:
                click.echo(f"Error: {msg}", err=True)
            sys.exit(1)

        # Block save on validation failure (MR5R2-C1)
        if not validation["valid"]:
            msg = f"Edit produced invalid scheme: {'; '.join(validation.get('issues', []))}"
            if json_output:
                click.echo(json.dumps({"status": "error", "error": msg, "validation": validation}))
            else:
                click.echo(f"Error: {msg}", err=True)
            sys.exit(1)

        save_scheme(result, scheme_path)

        if json_output:
            click.echo(
                json.dumps(
                    {"status": "ok", "action": "remove_step", "validation": validation},
                    ensure_ascii=False,
                )
            )
        else:
            click.echo(f"Removed step '{remove_step_id}' from {scheme_path}")
            if not validation["valid"]:
                click.echo("Warning: validation issues found after edit:", err=True)
                for issue in validation.get("issues", []):
                    click.echo(f"  - {issue}", err=True)
    else:
        msg = "Specify --add-step or --remove-step"
        if json_output:
            click.echo(json.dumps({"status": "error", "error": msg}))
        else:
            click.echo(f"Error: {msg}", err=True)
        sys.exit(1)


@main.command(name="export")
@click.argument("scheme_file", type=click.Path(exists=True))
@click.option("-o", "--output", required=True, type=click.Path(), help="Output file (.cdxml)")
@click.option("--force", is_flag=True, help="Overwrite existing output file")
@click.option("--json", "json_output", is_flag=True, help="Output JSON result instead of text")
def export_cdxml(scheme_file: str, output: str, force: bool, json_output: bool) -> None:
    """Export scheme.json to CDXML format for ChemDraw."""
    from chemscheme.renderers.cdxml_writer import scheme_to_cdxml
    from chemscheme.schema.scheme_schema import load_scheme

    scheme_path = Path(scheme_file)
    out_path = Path(output)

    if out_path.exists() and not force:
        msg = f"Output file already exists: {out_path}. Use --force to overwrite."
        if json_output:
            click.echo(json.dumps({"status": "error", "error": msg}))
        else:
            click.echo(f"Error: {msg}", err=True)
        sys.exit(1)

    # Validate output extension (MR4-2)
    if out_path.suffix.lower() != ".cdxml":
        msg = f"Unsupported output format '{out_path.suffix}'. Use .cdxml."
        if json_output:
            click.echo(json.dumps({"status": "error", "error": msg}))
        else:
            click.echo(f"Error: {msg}", err=True)
        sys.exit(1)

    try:
        scheme = load_scheme(scheme_path)
    except ValueError as e:  # MR5-C3: catch malformed JSON
        if json_output:
            click.echo(json.dumps({"status": "error", "error": str(e)}))
        else:
            click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # Validate scheme before export (MR4-8)
    from chemscheme.schema.scheme_schema import validate_scheme as _validate_schema

    schema_errors = _validate_schema(scheme)
    if schema_errors:
        msg = f"Scheme validation failed: {'; '.join(schema_errors)}"
        if json_output:
            click.echo(json.dumps({"status": "error", "error": msg, "issues": schema_errors}))
        else:
            click.echo(f"Error: {msg}", err=True)
        sys.exit(1)

    try:
        cdxml = scheme_to_cdxml(scheme)
    except (ValueError, RuntimeError) as e:  # MR4.1-P1e
        msg = f"CDXML export failed: {e}"
        if json_output:
            click.echo(json.dumps({"status": "error", "error": msg}))
        else:
            click.echo(f"Error: {msg}", err=True)
        sys.exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(cdxml, encoding="utf-8")

    if json_output:
        click.echo(
            json.dumps(
                {
                    "status": "ok",
                    "output_path": str(out_path),
                    "format": "cdxml",
                    "size_bytes": out_path.stat().st_size,
                },
                ensure_ascii=False,
            )
        )
    else:
        click.echo(f"Exported CDXML to {out_path} ({out_path.stat().st_size} bytes)")


@main.command(name="import")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", required=True, type=click.Path(), help="Output scheme.json path")
@click.option("--force", is_flag=True, help="Overwrite existing output file")
@click.option("--json", "json_output", is_flag=True, help="Output JSON result instead of text")
def import_structure(input_file: str, output: str, force: bool, json_output: bool) -> None:
    """Import a CDXML or CDX file to scheme.json format."""
    from chemscheme.renderers.cdxml_writer import cdxml_to_scheme
    from chemscheme.schema.scheme_schema import save_scheme, validate_scheme_json

    input_path = Path(input_file)
    out_path = Path(output)

    if out_path.exists() and not force:
        msg = f"Output file already exists: {out_path}. Use --force to overwrite."
        if json_output:
            click.echo(json.dumps({"status": "error", "error": msg}))
        else:
            click.echo(f"Error: {msg}", err=True)
        sys.exit(1)

    suffix = input_path.suffix.lower()

    if suffix == ".cdx":
        # CDX binary → convert to molecules, create single-step scheme
        from chemscheme.parsers.structure_reader import read_cdx

        try:
            results = read_cdx(input_path)
        except ValueError as e:
            msg = str(e)
            if json_output:
                click.echo(json.dumps({"status": "error", "error": msg}))
            else:
                click.echo(f"Error: {msg}", err=True)
            sys.exit(1)

        valid_smiles = [r.smiles for r in results if r.smiles]
        if not valid_smiles:
            msg = "No valid molecules found in CDX file"
            if json_output:
                click.echo(json.dumps({"status": "error", "error": msg}))
            else:
                click.echo(f"Error: {msg}", err=True)
            sys.exit(1)

        # AC-9.3: CDX has no reaction semantics — place molecules as reactants only
        scheme: dict = {  # type: ignore[type-arg]
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": valid_smiles,
                    "products": [],
                    "arrow_above": "",
                }
            ],
            "import_warnings": [
                "CDX binary import: molecules placed as reactants only; "
                "edit scheme to assign products and define reaction steps"
            ],
        }
    elif suffix == ".cdxml":
        try:
            cdxml_str = input_path.read_text(encoding="utf-8")  # C8 fix: inside try
            scheme = cdxml_to_scheme(cdxml_str)
        except Exception as e:  # ValueError, XML parse, UnicodeDecodeError (MR4-5, C8)
            msg = str(e)
            if json_output:
                click.echo(json.dumps({"status": "error", "error": msg}))
            else:
                click.echo(f"Error: {msg}", err=True)
            sys.exit(1)
    else:
        msg = f"Unsupported import format '{suffix}'. Use .cdxml or .cdx."
        if json_output:
            click.echo(json.dumps({"status": "error", "error": msg}))
        else:
            click.echo(f"Error: {msg}", err=True)
        sys.exit(1)

    # Validate imported scheme before saving (MR4-3)
    validation = validate_scheme_json(scheme)
    # Merge CDXML-specific import warnings with validation warnings (MR4.1-P2d)
    cdxml_warnings = scheme.pop("import_warnings", [])
    import_warnings = cdxml_warnings + validation.get("warnings", [])

    # Block save on validation failure (MR5R2-C2)
    # AC-9.3: CDX imports are partial (no reaction semantics) — save with warnings
    if not validation["valid"] and suffix != ".cdx":
        msg = f"Imported scheme is invalid: {'; '.join(validation.get('issues', []))}"
        if json_output:
            click.echo(
                json.dumps(
                    {
                        "status": "error",
                        "error": msg,
                        "validation": validation,
                        "import_warnings": cdxml_warnings,
                    }
                )
            )
        else:
            click.echo(f"Error: {msg}", err=True)
            for w in import_warnings:
                click.echo(f"  ⚠ {w}", err=True)
        sys.exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_scheme(scheme, out_path)

    if json_output:
        click.echo(
            json.dumps(
                {
                    "status": "ok",
                    "output_path": str(out_path),
                    "steps": len(scheme.get("steps", [])),
                    "validation": validation,
                    "import_warnings": cdxml_warnings,
                },
                ensure_ascii=False,
            )
        )
    else:
        steps = len(scheme.get("steps", []))
        click.echo(f"Imported {steps} step(s) to {out_path}")
        if not validation["valid"]:
            click.echo("Warning: imported scheme has validation issues:", err=True)
            for issue in validation.get("issues", []):
                click.echo(f"  - {issue}", err=True)
        for w in import_warnings:
            click.echo(f"  ⚠ {w}", err=True)


@main.command()
@click.argument("structure_file", type=click.Path(exists=True))
@click.option("--json", "json_output", is_flag=True, help="Output JSON result instead of text")
def read(structure_file: str, json_output: bool) -> None:
    """Read SMILES from a structure file (MOL, SDF, CDX)."""
    from chemscheme.parsers.structure_reader import read_structure

    structure_path = Path(structure_file)

    try:
        results = read_structure(structure_path)
    except ValueError as e:
        if json_output:
            click.echo(json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False))
        else:
            click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # Filter out failed molecules
    valid = [r for r in results if r.smiles]
    if not valid:
        msg = "No valid molecules found in file"
        if json_output:
            click.echo(json.dumps({"status": "error", "error": msg}, ensure_ascii=False))
        else:
            click.echo(f"Error: {msg}", err=True)
        sys.exit(1)

    if json_output:
        molecules = [
            {"smiles": r.smiles, "confidence": r.confidence, "source": r.source} for r in valid
        ]
        click.echo(
            json.dumps(
                {"status": "ok", "molecules": molecules, "count": len(molecules)},
                ensure_ascii=False,
            )
        )
    else:
        for r in valid:
            click.echo(r.smiles)


# ---------------------------------------------------------------------------
# Phase 18a: chemscheme extract — image → SMILES recognition
# ---------------------------------------------------------------------------


@main.command()
@click.argument("image_file", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), help="Output JSON file")
@click.option("--json", "json_output", is_flag=True, help="Output JSON to stdout")
@click.option("--force", is_flag=True, help="Overwrite existing output file")
def extract(image_file: str, output: str | None, json_output: bool, force: bool) -> None:
    """Recognize molecule structure from an image file (PNG/JPEG)."""
    from chemscheme.recognition.mocks import MockMoleculeRecognizer
    from chemscheme.recognition.ocsr import recognize_molecule

    image_path = Path(image_file)
    out_path = Path(output) if output else None

    # Overwrite protection
    if out_path and out_path.exists() and not force:
        msg = f"Output file already exists: {out_path}. Use --force to overwrite."
        if json_output:
            click.echo(json.dumps({"status": "error", "error": msg}, ensure_ascii=False))
        else:
            click.echo(f"Error: {msg}", err=True)
        sys.exit(1)

    # Read image bytes
    try:
        image_bytes = image_path.read_bytes()
    except Exception as e:
        msg = f"Failed to read image: {e}"
        if json_output:
            click.echo(json.dumps({"status": "error", "error": msg}, ensure_ascii=False))
        else:
            click.echo(f"Error: {msg}", err=True)
        sys.exit(1)

    if not image_bytes:
        msg = "Image file is empty"
        if json_output:
            click.echo(json.dumps({"status": "error", "error": msg}, ensure_ascii=False))
        else:
            click.echo(f"Error: {msg}", err=True)
        sys.exit(1)

    # Build recognizer list: try real Docker/subprocess tools, mock fallback
    import contextlib

    from chemscheme.recognition.ocsr import (
        DecimerRecognizer,
        MolNexTRRecognizer,
        MolScribeRecognizer,
    )

    # R2-02 fix: verify tools actually work before adding
    from chemscheme.recognition.runner import is_tool_available

    recognizers: list[MockMoleculeRecognizer] = []
    with contextlib.suppress(Exception):
        rec = MolScribeRecognizer()
        if rec._use_docker or is_tool_available(rec._cmd):  # noqa: SLF001
            recognizers.append(rec)  # type: ignore[arg-type]
    with contextlib.suppress(Exception):
        rec = DecimerRecognizer()  # type: ignore[assignment]
        if rec._use_docker or is_tool_available(rec._cmd):  # noqa: SLF001
            recognizers.append(rec)  # type: ignore[arg-type]
    # S68: MolNexTR (MARCUS-spec parity, Phase 17c) — highest single-tool accuracy
    with contextlib.suppress(Exception):
        rec = MolNexTRRecognizer()  # type: ignore[assignment]
        if rec._use_docker or is_tool_available(rec._cmd):  # noqa: SLF001
            recognizers.append(rec)  # type: ignore[arg-type]
    if not recognizers:
        click.echo("Warning: No real OCSR recognizers available, using mock", err=True)
        recognizers.append(MockMoleculeRecognizer())

    # Extract image format from file extension (C3-R2 fix)
    _ext = image_path.suffix.lstrip(".").lower()
    _img_fmt = "jpeg" if _ext in ("jpg", "jpeg") else (_ext or "png")

    try:
        result = recognize_molecule(
            image=image_bytes,
            recognizers=recognizers,  # type: ignore[arg-type]
            image_format=_img_fmt,
        )
    except Exception as e:
        msg = f"Recognition failed: {e}"
        if json_output:
            click.echo(json.dumps({"status": "error", "error": msg}, ensure_ascii=False))
        else:
            click.echo(f"Error: {msg}", err=True)
        sys.exit(1)

    # A4 fix: empty SMILES = recognition failure
    if not result.value:
        msg = "Recognition produced no result (needs human review)"
        if json_output:
            click.echo(json.dumps({"status": "error", "error": msg}, ensure_ascii=False))
        else:
            click.echo(f"Error: {msg}", err=True)
        sys.exit(1)

    result_data = {
        "status": "ok",
        "smiles": result.value,
        "confidence": result.confidence,
        "source": result.source,
        "agreement": (
            result.agreement.value if hasattr(result.agreement, "value") else str(result.agreement)
        ),
    }

    if out_path:
        # Auto-create parent directory
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result_data, ensure_ascii=False, indent=2), encoding="utf-8")
        if json_output:
            click.echo(json.dumps(result_data, ensure_ascii=False))
        else:
            click.echo(f"Wrote result to {out_path}")
            click.echo(f"SMILES: {result.value} (confidence: {result.confidence:.2f})")
    elif json_output:
        click.echo(json.dumps(result_data, ensure_ascii=False))
    else:
        click.echo(f"SMILES: {result.value}")
        click.echo(f"Confidence: {result.confidence:.2f}")
        click.echo(f"Source: {result.source}")


# ---------------------------------------------------------------------------
# Phase 18a: chemscheme extract-pdf — PDF → figure extraction + classification
# ---------------------------------------------------------------------------


@main.command(name="extract-pdf")
@click.argument("pdf_file", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), help="Output JSON file")
@click.option("--json", "json_output", is_flag=True, help="Output JSON to stdout")
@click.option("--force", is_flag=True, help="Overwrite existing output file")
def extract_pdf(pdf_file: str, output: str | None, json_output: bool, force: bool) -> None:
    """Extract and classify figures from a PDF file."""
    from chemscheme.recognition.pdf_extract import (
        classify_figure_heuristic,
        extract_figures,
    )

    pdf_path = Path(pdf_file)
    out_path = Path(output) if output else None

    # Directory target rejection
    if out_path and out_path.is_dir():
        msg = f"Output path is a directory: {out_path}. Provide a file path."
        if json_output:
            click.echo(json.dumps({"status": "error", "error": msg}, ensure_ascii=False))
        else:
            click.echo(f"Error: {msg}", err=True)
        sys.exit(1)

    # Overwrite protection
    if out_path and out_path.exists() and not force:
        msg = f"Output file already exists: {out_path}. Use --force to overwrite."
        if json_output:
            click.echo(json.dumps({"status": "error", "error": msg}, ensure_ascii=False))
        else:
            click.echo(f"Error: {msg}", err=True)
        sys.exit(1)

    try:
        figures = extract_figures(pdf_path)
    except FileNotFoundError as e:
        msg = str(e)
        if json_output:
            click.echo(json.dumps({"status": "error", "error": msg}, ensure_ascii=False))
        else:
            click.echo(f"Error: {msg}", err=True)
        sys.exit(1)
    except Exception as e:
        msg = f"PDF extraction failed: {e}"
        if json_output:
            click.echo(json.dumps({"status": "error", "error": msg}, ensure_ascii=False))
        else:
            click.echo(f"Error: {msg}", err=True)
        sys.exit(1)

    # Classify each figure
    classified: list[dict[str, int | str]] = []
    for fig in figures:
        fig_class = classify_figure_heuristic(fig.width, fig.height)
        classified.append(
            {
                "page": fig.page,
                "width": fig.width,
                "height": fig.height,
                "class": fig_class.value,
            }
        )

    result_data = {
        "status": "ok",
        "figures": classified,
        "total": len(classified),
    }

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result_data, ensure_ascii=False, indent=2), encoding="utf-8")
        if json_output:
            click.echo(json.dumps(result_data, ensure_ascii=False))
        else:
            click.echo(f"Wrote {len(classified)} figures to {out_path}")
    elif json_output:
        click.echo(json.dumps(result_data, ensure_ascii=False, indent=2))
    else:
        click.echo(f"Found {len(classified)} figures in {pdf_path.name}")
        for i, fig_info in enumerate(classified, 1):
            click.echo(
                f"  [{i}] Page {fig_info['page']}, "
                f"{fig_info['width']}x{fig_info['height']}px, "
                f"class: {fig_info['class']}"
            )


# ---------------------------------------------------------------------------
# Phase 19: chemscheme lookup — search reagent/abbreviation library
# ---------------------------------------------------------------------------


@main.command()
@click.argument("query")
@click.option(
    "--category", type=str, default=None, help="Filter by category (catalyst, base, solvent, ...)"
)
@click.option("--json", "json_output", is_flag=True, help="Output JSON to stdout")
def lookup(query: str, category: str | None, json_output: bool) -> None:
    """Search the reagent abbreviation library."""
    from chemscheme.domain_library import lookup_reagent

    results = lookup_reagent(query, category=category)

    if json_output:
        data = {
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
        }
        click.echo(json.dumps(data, ensure_ascii=False, indent=2))
    elif not results:
        click.echo(f"No reagents found matching '{query}'.")
    else:
        for r in results:
            smiles_str = f"  SMILES: {r.smiles}" if r.smiles else ""
            click.echo(f"{r.abbreviation} — {r.name} [{r.category}]{smiles_str}")


# ---------------------------------------------------------------------------
# Phase 19: chemscheme suggest — suggest reaction conditions
# ---------------------------------------------------------------------------


@main.command()
@click.argument("reaction_type")
@click.option("--json", "json_output", is_flag=True, help="Output JSON to stdout")
def suggest(reaction_type: str, json_output: bool) -> None:
    """Suggest typical conditions for a named reaction."""
    from chemscheme.domain_library import suggest_conditions

    result = suggest_conditions(reaction_type)

    if json_output:
        if result:
            data = {
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
            }
        else:
            # Show all available reactions when no match
            from chemscheme.domain_library import _load_reactions

            all_names = [r.name for r in _load_reactions()]
            data = {
                "status": "not_found",
                "query": reaction_type,
                "available_reactions": all_names,
            }
        click.echo(json.dumps(data, ensure_ascii=False, indent=2))
    elif result:
        click.echo(f"=== {result.name} ===")
        if result.catalyst:
            click.echo(f"  Catalyst: {result.catalyst}")
        if result.solvents:
            click.echo(f"  Solvents: {', '.join(result.solvents)}")
        if result.bases:
            click.echo(f"  Base: {', '.join(result.bases)}")
        if result.temperature:
            click.echo(f"  Temperature: {result.temperature}")
        if result.time:
            click.echo(f"  Time: {result.time}")
        if result.atmosphere:
            click.echo(f"  Atmosphere: {result.atmosphere}")
        if result.description:
            click.echo(f"  {result.description}")
    else:
        from chemscheme.domain_library import _load_reactions

        all_names = [r.name for r in _load_reactions()]
        click.echo(f"Reaction not found: '{reaction_type}'")
        click.echo(f"Available reactions: {', '.join(all_names)}")


# ---------------------------------------------------------------------------
# User-extensible library (Post-Phase 20)
# ---------------------------------------------------------------------------


@main.group()
def library() -> None:
    """Manage custom reagent library (~/.chemscheme/)."""


@library.command(name="list")
@click.option("--custom", "custom_only", is_flag=True, help="Show only custom reagents")
@click.option("--builtin", "builtin_only", is_flag=True, help="Show only built-in reagents")
@click.option("--json", "json_output", is_flag=True, help="Output JSON to stdout")
def library_list(custom_only: bool, builtin_only: bool, json_output: bool) -> None:
    """List reagents in the library."""
    from chemscheme.domain_library import _load_reagents, list_user_reagents

    if custom_only:
        reagents = list_user_reagents()
        label = "custom"
    elif builtin_only:
        # Built-in only: load without user merge
        from chemscheme.domain_library import _load_data_text

        data = json.loads(_load_data_text("reagent_aliases.json"))
        abbrevs = data.get("abbreviations", {})
        from chemscheme.domain_library import ReagentInfo

        reagents = [
            ReagentInfo(
                abbreviation=a,
                name=info.get("name", a) if isinstance(info, dict) else str(info),
                category=info.get("category", "reagent") if isinstance(info, dict) else "reagent",
                smiles=info.get("smiles") if isinstance(info, dict) else None,
            )
            for a, info in abbrevs.items()
            if not a.startswith("_")
        ]
        label = "built-in"
    else:
        reagents = _load_reagents()
        label = "all"

    if json_output:
        data_out = {
            "status": "ok",
            "source": label,
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
        }
        click.echo(json.dumps(data_out, ensure_ascii=False, indent=2))
    elif not reagents:
        click.echo(f"No {label} reagents found.")
    else:
        click.echo(f"{len(reagents)} {label} reagent(s):")
        for r in reagents:
            smiles_str = f"  SMILES: {r.smiles}" if r.smiles else ""
            click.echo(f"  {r.abbreviation} — {r.name} [{r.category}]{smiles_str}")


@library.command(name="add")
@click.argument("abbreviation")
@click.option("--name", required=True, help="Full name of the reagent")
@click.option("--category", required=True, help="Category (catalyst, solvent, reagent, ...)")
@click.option("--smiles", default=None, help="SMILES string (validated via RDKit)")
@click.option("--json", "json_output", is_flag=True, help="Output JSON to stdout")
def library_add(
    abbreviation: str, name: str, category: str, smiles: str | None, json_output: bool
) -> None:
    """Add a custom reagent to ~/.chemscheme/."""
    from chemscheme.domain_library import add_user_reagent

    try:
        result = add_user_reagent(abbreviation, name, category, smiles=smiles)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from None

    if json_output:
        click.echo(
            json.dumps(
                {
                    "status": "ok",
                    "abbreviation": result.abbreviation,
                    "name": result.name,
                    "category": result.category,
                    "smiles": result.smiles,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        click.echo(f"Added: {result.abbreviation} — {result.name} [{result.category}]")


@library.command(name="remove")
@click.argument("abbreviation")
@click.option("--json", "json_output", is_flag=True, help="Output JSON to stdout")
def library_remove(abbreviation: str, json_output: bool) -> None:
    """Remove a custom reagent from ~/.chemscheme/."""
    from chemscheme.domain_library import remove_user_reagent

    removed = remove_user_reagent(abbreviation)
    if json_output:
        click.echo(
            json.dumps(
                {"status": "ok" if removed else "not_found", "abbreviation": abbreviation},
                ensure_ascii=False,
            )
        )
    elif removed:
        click.echo(f"Removed: {abbreviation}")
    else:
        click.echo(f"Not found in custom library: {abbreviation}")


# ---------------------------------------------------------------------------
# Text extraction (Phase 20)
# ---------------------------------------------------------------------------


@main.command(name="extract-text")
@click.option("--text", "text_input", default=None, help="Inline text to extract from")
@click.option("--no-ai", is_flag=True, help="Disable Claude API, use only regex + heuristics")
@click.option("--json", "json_output", is_flag=True, help="Output JSON to stdout")
@click.option("-o", "--output", default=None, type=click.Path(), help="Output scheme.json file")
@click.option("--force", is_flag=True, help="Overwrite existing output file")
@click.argument("input_file", required=False, type=click.Path(exists=True))
def extract_text(
    text_input: str | None,
    no_ai: bool,
    json_output: bool,
    output: str | None,
    force: bool,
    input_file: str | None,
) -> None:
    """Extract reaction data from experimental text into scheme.json."""
    from chemscheme.text_extraction import extract_reactions_from_text

    # Get text from either --text flag or input file
    if text_input:
        text = text_input
    elif input_file:
        text = Path(input_file).read_text(encoding="utf-8")
    else:
        click.echo("Error: provide --text or an input file", err=True)
        sys.exit(1)

    result = extract_reactions_from_text(text, use_ai=not no_ai)

    if output:
        out_path = Path(output)
        if out_path.exists() and not force:
            click.echo(f"Error: {output} exists (use --force to overwrite)", err=True)
            sys.exit(1)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        click.echo(f"Wrote {output} ({len(result['steps'])} steps)")
    elif json_output:
        click.echo(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        steps = result["steps"]
        if not steps:
            click.echo("No reactions found in text.")
        else:
            click.echo(f"Found {len(steps)} reaction step(s):")
            for step in steps:
                click.echo(f"  [{step['id']}]")
                if step.get("reactants"):
                    click.echo(f"    Reactants: {', '.join(step['reactants'])}")
                if step.get("products"):
                    click.echo(f"    Products: {', '.join(step['products'])}")
                if step.get("reagents"):
                    click.echo(f"    Reagents: {', '.join(step['reagents'])}")
                if step.get("arrow_above"):
                    click.echo(f"    Conditions: {step['arrow_above']}")
                if step.get("yield"):
                    click.echo(f"    Yield: {step['yield']}")
        for w in result.get("warnings", []):
            click.echo(f"  Warning: {w}", err=True)


# ---------------------------------------------------------------------------
# Procedure generation (Phase 21)
# ---------------------------------------------------------------------------


@main.command(name="generate-procedure")
@click.option(
    "--format",
    "fmt",
    default="text",
    type=click.Choice(["text", "json"]),
    help="Output format: text (default) or json",
)
@click.option("-o", "--output", default=None, type=click.Path(), help="Write output to file")
@click.argument("scheme_file", type=click.Path(exists=True))
def generate_procedure_cmd(
    fmt: str,
    output: str | None,
    scheme_file: str,
) -> None:
    """Generate experimental procedure text from a scheme.json file."""
    from chemscheme.procedure_generator import generate_procedure

    # C4 fix: handle malformed JSON gracefully
    try:
        scheme = json.loads(Path(scheme_file).read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        click.echo(f"Error: invalid JSON in {scheme_file}: {e}", err=True)
        sys.exit(1)

    try:
        result = generate_procedure(scheme)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if fmt == "json":
        data = {
            "text": result.text,
            "steps": [
                {"step_number": s.step_number, "text": s.text, "reaction_type": s.reaction_type}
                for s in result.steps
            ],
            "warnings": result.warnings,
        }
        out_text = json.dumps(data, ensure_ascii=False, indent=2)
    else:
        out_text = result.text
        if result.warnings:
            for w in result.warnings:
                click.echo(f"Warning: {w}", err=True)

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out_text, encoding="utf-8")
        click.echo(f"Wrote {output}")
    else:
        click.echo(out_text)


@main.command("benchmark-ocsr")
@click.option(
    "--input-dir",
    "input_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Directory containing OCSR test images (PNG/JPEG).",
)
@click.option(
    "--ground-truth",
    "ground_truth_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="JSON file mapping image_id (filename stem) → expected SMILES.",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(),
    help="Output JSON report path.",
)
@click.option(
    "--tools",
    "tools_csv",
    default="molscribe,decimer,molnextr,mathpix",
    show_default=True,
    help="Comma-separated tool subset to benchmark.",
)
@click.option(
    "--mock",
    is_flag=True,
    help="Use mock recognizers (no docker required) — for framework smoke testing only.",
)
def benchmark_ocsr(
    input_dir: str,
    ground_truth_path: str,
    output: str,
    tools_csv: str,
    mock: bool,
) -> None:
    """Benchmark OCSR tools against a ground-truth fixture (S68 Part 4).

    Closes S67 Part 5b: gives empirical per-tool SMILES-level accuracy +
    coverage + confidence calibration so the user can decide Mathpix
    keep/drop/reweight from evidence rather than marketing claims.

    Real-tool execution requires Docker daemon + per-tool images built.
    Use --mock for framework smoke testing without docker.
    """
    from chemscheme.recognition.benchmark import generate_report, run_benchmark

    in_dir = Path(input_dir)
    gt_path = Path(ground_truth_path)
    out_path = Path(output)

    # Load ground truth
    try:
        ground_truth: dict[str, str] = json.loads(gt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        click.echo(f"Error: invalid ground_truth JSON: {e}", err=True)
        sys.exit(1)
    if not isinstance(ground_truth, dict):
        click.echo("Error: ground_truth must be a JSON object {image_id: SMILES}", err=True)
        sys.exit(1)

    # Discover images by filename stem
    images: dict[str, Path] = {}
    for ext in (".png", ".jpg", ".jpeg"):
        for img in in_dir.glob(f"*{ext}"):
            images[img.stem] = img
    if not images:
        click.echo(f"Error: no PNG/JPEG images found in {in_dir}", err=True)
        sys.exit(1)

    # Build recognizer set
    requested_tools = {t.strip().lower() for t in tools_csv.split(",") if t.strip()}
    recognizers: dict = {}
    if mock:
        from chemscheme.recognition.mocks import MockMoleculeRecognizer

        for tool_name in requested_tools:
            recognizers[tool_name] = MockMoleculeRecognizer()
    else:
        from chemscheme.recognition.ocsr import (
            DecimerRecognizer,
            MathpixRecognizer,
            MolNexTRRecognizer,
            MolScribeRecognizer,
        )

        registry = {
            "molscribe": MolScribeRecognizer,
            "decimer": DecimerRecognizer,
            "molnextr": MolNexTRRecognizer,
            "mathpix": MathpixRecognizer,
        }
        for tool_name in requested_tools:
            if tool_name not in registry:
                click.echo(f"Warning: unknown tool {tool_name!r} — skipping", err=True)
                continue
            try:
                recognizers[tool_name] = registry[tool_name]()
            except Exception as e:
                click.echo(f"Warning: failed to construct {tool_name}: {e}", err=True)

    if not recognizers:
        click.echo("Error: no recognizers available — try --mock", err=True)
        sys.exit(1)

    # Run + report
    click.echo(f"Benchmarking {len(recognizers)} tool(s) on {len(images)} image(s)...")
    results = run_benchmark(images=images, ground_truth=ground_truth, recognizers=recognizers)
    report = generate_report(results)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    click.echo(f"Wrote report: {out_path}")
    for tool, m in report["per_tool"].items():
        click.echo(
            f"  {tool:12s}  acc={m['accuracy']:.2%}  cov={m['coverage']:.2%}  "
            f"errors={m['errors']}/{m['total']}"
        )


@main.command()
@click.option(
    "--port",
    default=8765,
    type=int,
    show_default=True,
    help="Port to bind for the local authoring server (default: 8765).",
)
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="Host interface to bind (default: 127.0.0.1, localhost only).",
)
def serve(port: int, host: str) -> None:
    """Start the local authoring web server (Ketcher + scheme.json bridge).

    Open http://localhost:PORT in your browser to draw, paste SMILES, or
    drag-drop a paper screenshot. Requires the 'web' extra:
    `pip install chemscheme[web]`.
    """
    import socket

    # Pre-check the port so we can give a clean error message before uvicorn's
    # noisier traceback. There is a small race between this close() and uvicorn's
    # bind, but production conflicts (another `chemscheme serve` already up) are
    # caught reliably.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as test_sock:
            test_sock.bind((host, port))
    except OSError as e:
        click.echo(
            f"Error: port {port} is already in use ({e}). Try `chemscheme serve --port <other>`.",
            err=True,
        )
        sys.exit(1)

    try:
        import uvicorn

        from chemscheme.web.app import create_app
    except ImportError as e:
        click.echo(
            f"Error: 'web' extra not installed ({e}). Run: pip install 'chemscheme[web]'",
            err=True,
        )
        sys.exit(1)

    app = create_app()
    click.echo(f"ChemScheme authoring server: http://{host}:{port}/")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
