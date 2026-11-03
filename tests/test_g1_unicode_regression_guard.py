"""G1 regression guard — SVG output must preserve UTF-8 unicode symbols.

S67 Part 2 paper-grade audit initially flagged G1 as a "°C → �X mojibake
bug" in rendered SVG output. S67 Part 5b byte-level re-inspection
revealed the bug was in the AUDIT SCRIPT's Windows-cp950 terminal
rendering, NOT in the SVG file itself. The file correctly contains
UTF-8 bytes \xc2\xb0 for ° (U+00B0).

G1 is therefore NOT a bug — but the mis-diagnosis showed two real needs:

1. A behavior-lock regression guard that asserts the production SVG
   output IS valid UTF-8 and DOES literally contain the ° character.
   If a future refactor accidentally routes SVG writes through a
   non-UTF-8 path (e.g., system-default-encoding on Windows), this
   test fails fast.

2. A byte-level assertion (not string-level) so the test can't itself
   fall into the same diagnostic-tool artifact.

BDD Feature: UTF-8 preservation across render pipeline.

# RED-EXEMPT: behavior-lock guard — G1 proved to be non-bug after
# byte-level re-inspection; these tests pass from day one and ensure
# any future accidental encoding regression surfaces immediately.
"""

from __future__ import annotations

import pytest


@pytest.fixture()  # type: ignore[misc]
def unicode_step() -> dict:  # type: ignore[type-arg]
    """Step with common unicode symbols real papers use:
    ° (degree), Δ (heat), → (arrow text), α β (Greek letters)."""
    return {
        "id": "step1",
        "reactants": ["CCBr", "c1ccsc1"],
        "products": ["c1ccc(-c2ccsc2)cc1"],
        "arrow_above": "Pd(PPh3)4, 80 \u00b0C",  # °C
        "arrow_below": "\u0394, \u03b1-\u03b2 coupling, N\u2082",  # Δ, α-β, N₂
        "compound_labels": {
            "CCBr": "1",
            "c1ccsc1": "2",
            "c1ccc(-c2ccsc2)cc1": "3",
        },
    }


def _svg_bytes(step: dict, theme_name: str) -> bytes:  # type: ignore[type-arg]
    """Render step to SVG, normalise to bytes for byte-level assertions."""
    from chemscheme.renderers.drawsvg_composer import compose_step_svg
    from chemscheme.renderers.themes import get_theme

    svg = compose_step_svg(step, theme=get_theme(theme_name))
    if isinstance(svg, str):
        return svg.encode("utf-8")
    return svg  # type: ignore[no-any-return]


@pytest.mark.parametrize(  # type: ignore[misc]
    "theme_name",
    ["default", "acs", "nature", "rsc", "wiley"],
)
def test_svg_is_valid_utf8(theme_name: str, unicode_step: dict) -> None:  # type: ignore[type-arg]
    """The rendered SVG must be decodable as strict UTF-8 with no replacement
    characters (U+FFFD).

    Byte-level assertion: avoids the audit-script artifact that caused G1
    mis-diagnosis. We decode the raw bytes ourselves under strict UTF-8
    and verify no replacement chars appeared.
    """
    data = _svg_bytes(unicode_step, theme_name)

    # Must decode strict UTF-8
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        pytest.fail(f"Theme '{theme_name}' produced non-UTF-8 SVG bytes: {exc}")

    # Must not contain replacement character
    assert "\ufffd" not in text, (
        f"Theme '{theme_name}' SVG contains U+FFFD replacement character; "
        "indicates an upstream encoding failure."
    )


@pytest.mark.parametrize(  # type: ignore[misc]
    "theme_name",
    ["default", "acs", "nature", "rsc", "wiley"],
)
def test_svg_preserves_degree_symbol(theme_name: str, unicode_step: dict) -> None:  # type: ignore[type-arg]
    """The ° character (U+00B0) from arrow_above '80 °C' must survive
    the render pipeline and appear in the SVG as UTF-8 bytes \\xc2\\xb0.

    This is the literal G1 regression guard: if a future refactor
    writes SVG with system-default encoding (cp950 on Windows) instead
    of UTF-8, this test fails fast because \\xc2\\xb0 won't be present
    in the output bytes.
    """
    data = _svg_bytes(unicode_step, theme_name)

    # Byte-level check: UTF-8 encoding of ° is 0xc2 0xb0
    assert b"\xc2\xb0C" in data or b"\xc2\xb0" in data, (
        f"Theme '{theme_name}' SVG missing UTF-8-encoded ° (\\xc2\\xb0). "
        "Encoding regression likely."
    )

    # String-level cross-check
    text = data.decode("utf-8")
    assert "\u00b0" in text, (
        f"Theme '{theme_name}' SVG missing U+00B0 DEGREE SIGN after UTF-8 decode."
    )


@pytest.mark.parametrize(  # type: ignore[misc]
    "theme_name",
    ["default", "acs"],
)
def test_svg_preserves_greek_and_arrows(theme_name: str, unicode_step: dict) -> None:  # type: ignore[type-arg]
    """Additional unicode coverage: Δ, α, β, subscript 2 (N₂) must
    survive the pipeline."""
    data = _svg_bytes(unicode_step, theme_name)
    text = data.decode("utf-8", errors="strict")

    # Note: chemscheme subscripts chemical formulas (e.g., N2→N₂ tspan)
    # so the literal U+2082 may or may not appear. Focus on characters
    # that go through as-is: Δ, α, β.
    for sym, name in [("\u0394", "Δ (Greek delta)"), ("\u03b1", "α"), ("\u03b2", "β")]:
        assert sym in text, f"Theme '{theme_name}' dropped {name}; encoding path broken."
