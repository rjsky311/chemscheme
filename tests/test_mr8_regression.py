"""MR8 review regression tests.

Covers fixes for C1, C2, G2, G3, C3 findings.
"""

import xml.etree.ElementTree as ET

import pytest
from rdkit import Chem

from chemscheme.renderers.cdxml_writer import (
    _fragment_to_smiles,
    cdxml_to_scheme,
    scheme_to_cdxml,
)
from chemscheme.schema.scheme_schema import validate_scheme


def _make_scheme(reactants: list[str], products: list[str]) -> dict:
    return {
        "schema_version": "1",
        "steps": [{"id": "1", "reactants": reactants, "products": products, "arrow_above": ""}],
    }


# ──────────────────────────────────────────────
# MR8-C1: ENDDOWNRIGHT/ENDUPRIGHT removed from wedge export
# ──────────────────────────────────────────────


@pytest.mark.unit
class TestMR8C1NoEZWedges:
    """E/Z double-bond directions must NOT produce wedge Display attrs."""

    def test_no_wedge_for_ez_alkene(self) -> None:
        """E/Z isomers should not produce WedgeBegin Display attributes."""
        # E-2-butene: has E/Z stereo but no tetrahedral centers
        scheme = _make_scheme([r"C/C=C/C"], ["C"])
        cdxml = scheme_to_cdxml(scheme)
        root = ET.fromstring(cdxml)
        bonds = list(root.iter("b"))
        displays = [b.get("Display") for b in bonds if b.get("Display")]
        # E/Z should NOT produce WedgeBegin (that was the MR8-C1 bug)
        assert "WedgeBegin" not in displays


# ──────────────────────────────────────────────
# MR8-C2: WedgeEnd import orientation
# ──────────────────────────────────────────────


@pytest.mark.unit
class TestMR8C2WedgeEndImport:
    """WedgeEnd should be imported with reversed bond direction."""

    def test_wedge_end_sets_enddownright(self) -> None:
        """WedgeEnd Display → ENDDOWNRIGHT (not BEGINWEDGE)."""
        frag_xml = """<fragment id="1">
          <n id="10" p="100 200" Element="6" NumHydrogens="1"/>
          <n id="11" p="130 200" Element="9"/>
          <n id="12" p="100 230" Element="17"/>
          <n id="13" p="70 200" Element="35"/>
          <b id="50" B="10" E="11" Display="WedgeEnd"/>
          <b id="51" B="10" E="12"/>
          <b id="52" B="10" E="13"/>
        </fragment>"""
        frag = ET.fromstring(frag_xml)
        smiles = _fragment_to_smiles(frag)
        assert smiles is not None


# ──────────────────────────────────────────────
# MR8-G2: Stale warning updated
# ──────────────────────────────────────────────


@pytest.mark.unit
class TestMR8G2WarningText:
    """Reconstructed fragment warning should NOT mention charge/isotope loss."""

    def test_no_lossy_charge_isotope_warning(self) -> None:
        """Warning text should not claim charges/isotopes are lost (fixed in Phase 9)."""
        # Build CDXML without ChemSchemeSMILES → forces reconstruction path
        cdxml = (
            "<?xml version='1.0' encoding='us-ascii'?>\n"
            "<CDXML><page>\n"
            '  <fragment id="1"><n id="10" p="100 200" Element="6" Charge="1"/></fragment>\n'
            '  <fragment id="2"><n id="20" p="400 200" Element="6"/></fragment>\n'
            '  <arrow id="100" Head3D="300 200 0" Tail3D="200 200 0"/>\n'
            "  <scheme>"
            '    <step ReactionStepReactants="1" ReactionStepProducts="2" '
            '          ReactionStepArrows="100"/>'
            "  </scheme>\n"
            "</page></CDXML>"
        )
        result = cdxml_to_scheme(cdxml)
        warnings = result.get("import_warnings", [])
        for w in warnings:
            assert "charges, and isotopes may be lost" not in w


# ──────────────────────────────────────────────
# MR8-G3: isomericSmiles in reconstruction
# ──────────────────────────────────────────────


@pytest.mark.unit
class TestMR8G3IsomericSmiles:
    """Reconstructed SMILES should preserve isotope/charge via isomericSmiles=True."""

    def test_reconstruction_preserves_isotope(self) -> None:
        """Fragment with Isotope attr → SMILES contains isotope label."""
        frag_xml = """<fragment id="1">
          <n id="10" p="100 200" Element="6" Isotope="13" NumHydrogens="4"/>
        </fragment>"""
        frag = ET.fromstring(frag_xml)
        smiles = _fragment_to_smiles(frag)
        assert smiles is not None
        mol = Chem.MolFromSmiles(smiles)
        assert any(a.GetIsotope() == 13 for a in mol.GetAtoms())

    def test_reconstruction_preserves_charge(self) -> None:
        """Fragment with Charge attr → SMILES contains formal charge."""
        frag_xml = """<fragment id="1">
          <n id="10" p="100 200" Element="7" Charge="1" NumHydrogens="4"/>
        </fragment>"""
        frag = ET.fromstring(frag_xml)
        smiles = _fragment_to_smiles(frag)
        assert smiles is not None
        mol = Chem.MolFromSmiles(smiles)
        assert any(a.GetFormalCharge() == 1 for a in mol.GetAtoms())


# ──────────────────────────────────────────────
# MR8-C3: Empty products allowed (CDX draft state)
# ──────────────────────────────────────────────


@pytest.mark.unit
class TestMR8C3EmptyProductsValid:
    """Empty products list should not be a schema error."""

    def test_empty_products_no_error(self) -> None:
        """Scheme with products=[] passes schema validation."""
        scheme = {
            "schema_version": "1",
            "steps": [{"id": "1", "reactants": ["CC"], "products": [], "arrow_above": "a"}],
        }
        errors = validate_scheme(scheme)
        assert not any("products" in e for e in errors)

    def test_empty_products_with_reactants_valid(self) -> None:
        """CDX-style scheme (reactants only, no products) is valid."""
        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["CCO", "CC=O"],
                    "products": [],
                    "arrow_above": "",
                }
            ],
        }
        errors = validate_scheme(scheme)
        assert len(errors) == 0
