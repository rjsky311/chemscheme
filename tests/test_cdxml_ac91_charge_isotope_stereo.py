"""Tests for AC-9.1: CDXML export preserves charge, isotope, and bond stereo.

Phase 9 — MR7-01 (P1).
BDD-first: all tests RED before implementation.
"""

import xml.etree.ElementTree as ET

import pytest
from rdkit import Chem

from chemscheme.renderers.cdxml_writer import scheme_to_cdxml


def _make_scheme(smiles_list: list[str]) -> dict:
    """Helper: build a minimal scheme dict with one step."""
    return {
        "schema_version": "1",
        "steps": [
            {
                "id": "1",
                "reactants": smiles_list,
                "products": ["C"],  # trivial product
                "arrow_above": "test",
            }
        ],
    }


def _get_fragment_nodes(cdxml_str: str) -> list[ET.Element]:
    """Parse CDXML and return all <n> (atom node) elements."""
    root = ET.fromstring(cdxml_str)
    return list(root.iter("n"))


def _get_fragment_bonds(cdxml_str: str) -> list[ET.Element]:
    """Parse CDXML and return all <b> (bond) elements."""
    root = ET.fromstring(cdxml_str)
    return list(root.iter("b"))


# ──────────────────────────────────────────────
# Scenario 1: Formal charge export
# ──────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.chemistry
class TestCdxmlExportCharge:
    """AC-9.1: CDXML export writes Charge attribute for charged atoms."""

    def test_positive_charge_ammonium(self) -> None:
        """[NH4+] should produce a node with Charge='1'."""
        scheme = _make_scheme(["[NH4+]"])
        cdxml = scheme_to_cdxml(scheme)
        nodes = _get_fragment_nodes(cdxml)

        # Find the nitrogen node (Element=7)
        n_nodes = [n for n in nodes if n.get("Element") == "7"]
        assert len(n_nodes) >= 1, "Should have at least one nitrogen node"
        assert n_nodes[0].get("Charge") == "1", "Ammonium nitrogen must have Charge='1'"

    def test_negative_charge_carboxylate(self) -> None:
        """[O-]C=O (formate) should produce a node with Charge='-1'."""
        scheme = _make_scheme(["[O-]C=O"])
        cdxml = scheme_to_cdxml(scheme)
        nodes = _get_fragment_nodes(cdxml)

        # Find oxygen nodes with negative charge
        o_nodes = [n for n in nodes if n.get("Element") == "8"]
        charged = [n for n in o_nodes if n.get("Charge") == "-1"]
        assert len(charged) >= 1, "Formate oxygen must have Charge='-1'"

    def test_zwitterion_both_charges(self) -> None:
        """Glycine zwitterion [NH3+]CC([O-])=O has both +1 and -1."""
        scheme = _make_scheme(["[NH3+]CC([O-])=O"])
        cdxml = scheme_to_cdxml(scheme)
        nodes = _get_fragment_nodes(cdxml)

        charges = [n.get("Charge") for n in nodes if n.get("Charge")]
        assert "1" in charges, "Should have a +1 charge"
        assert "-1" in charges, "Should have a -1 charge"

    def test_no_charge_no_attribute(self) -> None:
        """Uncharged molecule (ethanol CCO) should NOT have Charge attributes.

        Adversarial: ensure no spurious Charge='0' is written.
        """
        scheme = _make_scheme(["CCO"])
        cdxml = scheme_to_cdxml(scheme)
        nodes = _get_fragment_nodes(cdxml)

        for node in nodes:
            assert node.get("Charge") is None, (
                f"Uncharged atom should not have Charge attribute, got '{node.get('Charge')}'"
            )


# ──────────────────────────────────────────────
# Scenario 2: Isotope export
# ──────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.chemistry
class TestCdxmlExportIsotope:
    """AC-9.1: CDXML export writes Isotope attribute for isotope-labeled atoms."""

    def test_carbon_13_label(self) -> None:
        """[13CH4] should produce a node with Isotope='13'."""
        scheme = _make_scheme(["[13CH4]"])
        cdxml = scheme_to_cdxml(scheme)
        nodes = _get_fragment_nodes(cdxml)

        # Carbon-13: Element defaults to 6 (omitted), but must have Isotope
        isotope_nodes = [n for n in nodes if n.get("Isotope")]
        assert len(isotope_nodes) >= 1, "Should have at least one isotope-labeled node"
        assert isotope_nodes[0].get("Isotope") == "13"

    def test_deuterium_label(self) -> None:
        """[2H]O (deuterated water) should produce a node with Isotope='2'."""
        scheme = _make_scheme(["[2H]O"])
        cdxml = scheme_to_cdxml(scheme)
        nodes = _get_fragment_nodes(cdxml)

        h_nodes = [n for n in nodes if n.get("Element") == "1"]
        isotope_h = [n for n in h_nodes if n.get("Isotope") == "2"]
        assert len(isotope_h) >= 1, "Deuterium must have Isotope='2'"

    def test_no_isotope_no_attribute(self) -> None:
        """Normal molecule should NOT have Isotope attributes.

        Adversarial: ensure no spurious Isotope='0' or Isotope='12'.
        """
        scheme = _make_scheme(["CCO"])
        cdxml = scheme_to_cdxml(scheme)
        nodes = _get_fragment_nodes(cdxml)

        for node in nodes:
            assert node.get("Isotope") is None, (
                f"Non-isotope atom should not have Isotope attribute, got '{node.get('Isotope')}'"
            )


# ──────────────────────────────────────────────
# Scenario 3: Bond stereo export (wedge/hash)
# ──────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.chemistry
class TestCdxmlExportBondStereo:
    """AC-9.1: CDXML export writes Display attribute for stereo bonds."""

    def test_chiral_center_wedge_bonds(self) -> None:
        """F[C@@H](Cl)Br should produce at least one bond with Display attribute.

        RDKit assigns wedge/hash bond directions for chiral centers.
        CDXML uses Display='WedgeBegin' or Display='WedgedHashBegin'.
        """
        scheme = _make_scheme(["F[C@@H](Cl)Br"])
        cdxml = scheme_to_cdxml(scheme)
        bonds = _get_fragment_bonds(cdxml)

        display_bonds = [b for b in bonds if b.get("Display")]
        assert len(display_bonds) >= 1, "Chiral center should produce at least one wedge/hash bond"
        valid_displays = {"WedgeBegin", "WedgedHashBegin", "WedgeEnd", "WedgedHashEnd"}
        for b in display_bonds:
            assert b.get("Display") in valid_displays, (
                f"Unexpected Display value: {b.get('Display')}"
            )

    def test_no_stereo_no_display(self) -> None:
        """Achiral molecule (ethanol) should NOT have Display attributes.

        Adversarial: ensure no spurious stereo indicators.
        """
        scheme = _make_scheme(["CCO"])
        cdxml = scheme_to_cdxml(scheme)
        bonds = _get_fragment_bonds(cdxml)

        for bond in bonds:
            assert bond.get("Display") is None, (
                f"Achiral bond should not have Display attribute, got '{bond.get('Display')}'"
            )


# ──────────────────────────────────────────────
# Scenario 4: Combined charge + isotope + stereo
# ──────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.chemistry
class TestCdxmlExportCombined:
    """AC-9.1: Multiple chemical features preserved simultaneously."""

    def test_charged_isotope_molecule(self) -> None:
        """[13NH4+] has both isotope=13 and charge=+1 on nitrogen."""
        scheme = _make_scheme(["[13NH4+]"])
        cdxml = scheme_to_cdxml(scheme)
        nodes = _get_fragment_nodes(cdxml)

        n_nodes = [n for n in nodes if n.get("Element") == "7"]
        assert len(n_nodes) >= 1
        n_node = n_nodes[0]
        assert n_node.get("Charge") == "1", "Must have Charge='1'"
        assert n_node.get("Isotope") == "13", "Must have Isotope='13'"

    def test_stereo_with_charge(self) -> None:
        """Chiral charged molecule preserves both stereo and charge.

        [C@@H](F)(Cl)[NH3+] — chiral center + charged amine.
        """
        smiles = "[C@@H](F)(Cl)[NH3+]"
        # Verify SMILES is valid first
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"Test SMILES should be valid: {smiles}"

        scheme = _make_scheme([smiles])
        cdxml = scheme_to_cdxml(scheme)

        nodes = _get_fragment_nodes(cdxml)
        bonds = _get_fragment_bonds(cdxml)

        # Charge on nitrogen
        n_nodes = [n for n in nodes if n.get("Element") == "7"]
        assert any(n.get("Charge") == "1" for n in n_nodes), "Charged nitrogen must have Charge='1'"

        # Stereo bonds
        display_bonds = [b for b in bonds if b.get("Display")]
        assert len(display_bonds) >= 1, "Chiral center should produce stereo bond(s)"


# ──────────────────────────────────────────────
# Scenario 5: SMILES round-trip through CDXML preserves features
# ──────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.chemistry
class TestCdxmlExportSmilesPreservation:
    """The stored ChemSchemeSMILES must preserve charge/isotope/stereo."""

    def test_stored_smiles_preserves_charge(self) -> None:
        """ChemSchemeSMILES attribute retains charge info."""
        scheme = _make_scheme(["[NH4+]"])
        cdxml = scheme_to_cdxml(scheme)
        root = ET.fromstring(cdxml)

        fragments = list(root.iter("fragment"))
        # First fragment is the reactant
        stored = fragments[0].get("ChemSchemeSMILES", "")
        mol = Chem.MolFromSmiles(stored)
        assert mol is not None
        assert Chem.MolToSmiles(mol) == Chem.MolToSmiles(Chem.MolFromSmiles("[NH4+]"))

    def test_stored_smiles_preserves_isotope(self) -> None:
        """ChemSchemeSMILES attribute retains isotope info."""
        scheme = _make_scheme(["[13CH4]"])
        cdxml = scheme_to_cdxml(scheme)
        root = ET.fromstring(cdxml)

        fragments = list(root.iter("fragment"))
        stored = fragments[0].get("ChemSchemeSMILES", "")
        mol = Chem.MolFromSmiles(stored)
        assert mol is not None
        # The stored SMILES must round-trip isotope
        assert any(a.GetIsotope() == 13 for a in mol.GetAtoms()), (
            "Stored SMILES must preserve isotope labels"
        )

    def test_stored_smiles_preserves_stereo(self) -> None:
        """ChemSchemeSMILES attribute retains stereo info."""
        scheme = _make_scheme(["F[C@@H](Cl)Br"])
        cdxml = scheme_to_cdxml(scheme)
        root = ET.fromstring(cdxml)

        fragments = list(root.iter("fragment"))
        stored = fragments[0].get("ChemSchemeSMILES", "")
        mol = Chem.MolFromSmiles(stored)
        assert mol is not None
        # Must have chiral info
        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        assert len(chiral_centers) >= 1, "Stored SMILES must preserve chirality"
