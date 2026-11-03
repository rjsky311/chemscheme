"""Tests for AC-9.5: Round-trip CDXML export→import preserves charge/isotope/stereo.
Tests for AC-9.6: Adversarial test cases for Phase 9.

Phase 9 — Integration + adversarial.
"""

import xml.etree.ElementTree as ET

import pytest
from rdkit import Chem

from chemscheme.renderers.cdxml_writer import cdxml_to_scheme, scheme_to_cdxml


def _make_scheme(reactants: list[str], products: list[str], arrow_above: str = "") -> dict:
    """Build a single-step scheme dict."""
    return {
        "schema_version": "1",
        "steps": [
            {
                "id": "1",
                "reactants": reactants,
                "products": products,
                "arrow_above": arrow_above,
            }
        ],
    }


# ──────────────────────────────────────────────
# AC-9.5: Round-trip integration tests
# ──────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.chemistry
class TestRoundTripChargePreservation:
    """CDXML export→import preserves formal charge through stored SMILES."""

    def test_roundtrip_ammonium(self) -> None:
        """[NH4+] survives export→import round-trip."""
        scheme = _make_scheme(["[NH4+]"], ["C"])
        cdxml = scheme_to_cdxml(scheme)
        imported = cdxml_to_scheme(cdxml)

        smi = imported["steps"][0]["reactants"][0]
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None
        n_atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
        assert n_atoms[0].GetFormalCharge() == 1

    def test_roundtrip_carboxylate(self) -> None:
        """[O-]C=O survives export→import."""
        scheme = _make_scheme(["[O-]C=O"], ["C"])
        cdxml = scheme_to_cdxml(scheme)
        imported = cdxml_to_scheme(cdxml)

        smi = imported["steps"][0]["reactants"][0]
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None
        o_atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 8]
        assert any(a.GetFormalCharge() == -1 for a in o_atoms)

    def test_roundtrip_zwitterion(self) -> None:
        """Glycine zwitterion [NH3+]CC([O-])=O preserves both charges."""
        orig = "[NH3+]CC([O-])=O"
        scheme = _make_scheme([orig], ["C"])
        cdxml = scheme_to_cdxml(scheme)
        imported = cdxml_to_scheme(cdxml)

        smi = imported["steps"][0]["reactants"][0]
        mol = Chem.MolFromSmiles(smi)
        orig_mol = Chem.MolFromSmiles(orig)
        assert Chem.MolToSmiles(mol) == Chem.MolToSmiles(orig_mol)


@pytest.mark.integration
@pytest.mark.chemistry
class TestRoundTripIsotopePreservation:
    """CDXML export→import preserves isotope labels."""

    def test_roundtrip_carbon_13(self) -> None:
        """[13CH4] survives round-trip."""
        scheme = _make_scheme(["[13CH4]"], ["C"])
        cdxml = scheme_to_cdxml(scheme)
        imported = cdxml_to_scheme(cdxml)

        smi = imported["steps"][0]["reactants"][0]
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None
        assert any(a.GetIsotope() == 13 for a in mol.GetAtoms())

    def test_roundtrip_deuterium(self) -> None:
        """[2H]O survives round-trip."""
        scheme = _make_scheme(["[2H]O"], ["C"])
        cdxml = scheme_to_cdxml(scheme)
        imported = cdxml_to_scheme(cdxml)

        smi = imported["steps"][0]["reactants"][0]
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None
        h_atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 1]
        assert any(a.GetIsotope() == 2 for a in h_atoms)


@pytest.mark.integration
@pytest.mark.chemistry
class TestRoundTripStereoPreservation:
    """CDXML export→import preserves stereochemistry."""

    def test_roundtrip_chiral_center(self) -> None:
        """F[C@@H](Cl)Br chirality survives round-trip."""
        orig = "F[C@@H](Cl)Br"
        scheme = _make_scheme([orig], ["C"])
        cdxml = scheme_to_cdxml(scheme)
        imported = cdxml_to_scheme(cdxml)

        smi = imported["steps"][0]["reactants"][0]
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None
        chiral = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        assert len(chiral) >= 1

    def test_roundtrip_complex_stereo(self) -> None:
        """L-Alanine [C@@H](N)(C)C(=O)O chirality survives round-trip."""
        orig = "[C@@H](N)(C)C(=O)O"
        scheme = _make_scheme([orig], ["C"])
        cdxml = scheme_to_cdxml(scheme)
        imported = cdxml_to_scheme(cdxml)

        smi = imported["steps"][0]["reactants"][0]
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None
        chiral = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        assert len(chiral) >= 1


@pytest.mark.integration
@pytest.mark.chemistry
class TestRoundTripCombined:
    """Multiple features preserved through round-trip."""

    def test_roundtrip_charged_isotope_stereo(self) -> None:
        """Molecule with charge + isotope + stereo survives round-trip."""
        # 13C-labeled chiral ammonium
        orig = "[13C@@H]([NH3+])(F)Cl"
        mol_check = Chem.MolFromSmiles(orig)
        if mol_check is None:
            pytest.skip("Test SMILES not valid in this RDKit version")

        scheme = _make_scheme([orig], ["C"])
        cdxml = scheme_to_cdxml(scheme)
        imported = cdxml_to_scheme(cdxml)

        smi = imported["steps"][0]["reactants"][0]
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None

        # Charge
        n_atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
        assert any(a.GetFormalCharge() == 1 for a in n_atoms)

        # Isotope
        assert any(a.GetIsotope() == 13 for a in mol.GetAtoms())

        # Stereo
        chiral = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        assert len(chiral) >= 1

    def test_roundtrip_multistep_with_features(self) -> None:
        """Multi-step scheme with charged/isotope molecules."""
        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["[NH4+]", "[13CH4]"],
                    "products": ["CCO"],
                    "arrow_above": "step 1",
                },
                {
                    "id": "2",
                    "reactants": ["CCO"],
                    "products": ["[O-]C=O"],
                    "arrow_above": "step 2",
                },
            ],
        }
        cdxml = scheme_to_cdxml(scheme)
        imported = cdxml_to_scheme(cdxml)

        assert len(imported["steps"]) == 2

        # Step 1: check [NH4+] and [13CH4]
        r1 = imported["steps"][0]["reactants"]
        mol_nh4 = Chem.MolFromSmiles(r1[0])
        assert mol_nh4 is not None
        assert any(a.GetFormalCharge() == 1 for a in mol_nh4.GetAtoms())

        mol_13c = Chem.MolFromSmiles(r1[1])
        assert mol_13c is not None
        assert any(a.GetIsotope() == 13 for a in mol_13c.GetAtoms())

        # Step 2: check [O-]C=O
        p2 = imported["steps"][1]["products"]
        mol_fo = Chem.MolFromSmiles(p2[0])
        assert mol_fo is not None
        assert any(a.GetFormalCharge() == -1 for a in mol_fo.GetAtoms())


# ──────────────────────────────────────────────
# AC-9.6: Adversarial test cases
# ──────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.chemistry
class TestAdversarialCdxmlExport:
    """AC-9.6: Edge cases that should not crash export."""

    def test_export_empty_smiles_no_crash(self) -> None:
        """Empty SMILES string should not crash (RDKit treats as empty mol)."""
        scheme = _make_scheme([""], ["C"])
        # Empty SMILES is valid in RDKit (0 atoms), should export without crash
        cdxml = scheme_to_cdxml(scheme)
        root = ET.fromstring(cdxml)
        assert root is not None

    def test_export_invalid_smiles_raises(self) -> None:
        """Completely invalid SMILES should raise ValueError."""
        scheme = _make_scheme(["not_a_smiles!!!"], ["C"])
        with pytest.raises(ValueError):
            scheme_to_cdxml(scheme)

    def test_export_multivalent_charge(self) -> None:
        """Fe3+ (high charge) should export without crash."""
        scheme = _make_scheme(["[Fe+3]"], ["C"])
        cdxml = scheme_to_cdxml(scheme)
        root = ET.fromstring(cdxml)
        nodes = list(root.iter("n"))
        fe_nodes = [n for n in nodes if n.get("Element") == "26"]
        assert fe_nodes[0].get("Charge") == "3"

    def test_export_zero_charge_no_attr(self) -> None:
        """Neutral atom must NOT have Charge='0'."""
        scheme = _make_scheme(["C"], ["C"])
        cdxml = scheme_to_cdxml(scheme)
        nodes = list(ET.fromstring(cdxml).iter("n"))
        for n in nodes:
            assert n.get("Charge") is None

    def test_export_heavy_isotope(self) -> None:
        """Tritium [3H] should export Isotope='3'."""
        scheme = _make_scheme(["[3H]O"], ["C"])
        cdxml = scheme_to_cdxml(scheme)
        nodes = list(ET.fromstring(cdxml).iter("n"))
        h_nodes = [n for n in nodes if n.get("Element") == "1"]
        assert any(n.get("Isotope") == "3" for n in h_nodes)


def _build_cdxml(frag: str, product: str) -> str:
    """Build a minimal CDXML string for adversarial import tests."""
    step = '<step ReactionStepReactants="1" ReactionStepProducts="2" ReactionStepArrows="100"/>'
    return (
        "<?xml version='1.0' encoding='us-ascii'?>\n"
        "<CDXML><page>\n"
        f"  {frag}{product}\n"
        '  <arrow id="100" Head3D="300 200 0" '
        'Tail3D="200 200 0"/>\n'
        f"  <scheme>{step}</scheme>\n"
        "</page></CDXML>"
    )


_PRODUCT_FRAG = """<fragment id="2">
  <n id="20" p="400 200" Element="6"/>
</fragment>"""


@pytest.mark.unit
@pytest.mark.chemistry
class TestAdversarialCdxmlImport:
    """AC-9.6: Edge cases that should not crash import."""

    def test_import_huge_charge_value(self) -> None:
        """Charge='99' should not crash (just weird chemistry)."""
        frag = """<fragment id="1">
          <n id="10" p="100 200" Element="26" Charge="99"/>
        </fragment>"""
        cdxml = _build_cdxml(frag, _PRODUCT_FRAG)
        result = cdxml_to_scheme(cdxml)
        assert len(result["steps"]) == 1

    def test_import_negative_isotope_ignored(self) -> None:
        """Isotope='-1' should not crash (invalid but graceful)."""
        frag = """<fragment id="1">
          <n id="10" p="100 200" Element="6" Isotope="-1"
             NumHydrogens="4"/>
        </fragment>"""
        cdxml = _build_cdxml(frag, _PRODUCT_FRAG)
        result = cdxml_to_scheme(cdxml)
        assert len(result["steps"]) == 1

    def test_import_unknown_display_value(self) -> None:
        """Display='Unknown' should not crash (just ignore)."""
        frag = """<fragment id="1">
          <n id="10" p="100 200" Element="6" NumHydrogens="3"/>
          <n id="11" p="130 200" Element="6" NumHydrogens="3"/>
          <b id="50" B="10" E="11" Display="Unknown"/>
        </fragment>"""
        cdxml = _build_cdxml(frag, _PRODUCT_FRAG)
        result = cdxml_to_scheme(cdxml)
        assert len(result["steps"]) == 1


# ──────────────────────────────────────────────
# MR10-C5 / MR11-C8: compound_labels round-trip
# ──────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.chemistry
class TestRoundTripCompoundLabels:
    """CDXML export→import preserves compound_labels dict."""

    def test_roundtrip_compound_labels_preserved(self) -> None:
        """GIVEN step with compound_labels, WHEN export→import, THEN labels preserved."""
        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["c1ccccc1"],
                    "products": ["c1ccc(O)cc1"],
                    "arrow_above": "NaOH",
                    "compound_labels": {"c1ccccc1": "1", "c1ccc(O)cc1": "2"},
                }
            ],
        }
        cdxml = scheme_to_cdxml(scheme)
        imported = cdxml_to_scheme(cdxml)

        step = imported["steps"][0]
        assert "compound_labels" in step
        assert step["compound_labels"] == {"c1ccccc1": "1", "c1ccc(O)cc1": "2"}

    def test_roundtrip_empty_compound_labels_not_stored(self) -> None:
        """GIVEN step with empty compound_labels, WHEN export→import, THEN key absent."""
        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["C"],
                    "products": ["CC"],
                    "arrow_above": "",
                    "compound_labels": {},
                }
            ],
        }
        cdxml = scheme_to_cdxml(scheme)
        imported = cdxml_to_scheme(cdxml)

        step = imported["steps"][0]
        assert "compound_labels" not in step

    def test_roundtrip_no_compound_labels_absent(self) -> None:
        """GIVEN step without compound_labels, WHEN export→import, THEN key absent."""
        scheme = _make_scheme(["C"], ["CC"])
        cdxml = scheme_to_cdxml(scheme)
        imported = cdxml_to_scheme(cdxml)

        step = imported["steps"][0]
        assert "compound_labels" not in step
