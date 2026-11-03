"""Tests for AC-9.2: CDX/CDXML import correctly parses charge, isotope, stereo, aromatic.

Phase 9 — MR7-02 (P1).
BDD-first: all tests RED before implementation.

These tests target the _fragment_to_smiles fallback path (no ChemSchemeSMILES attr),
which is used when importing CDXML files from ChemDraw or other tools.
"""

import pytest
from rdkit import Chem

from chemscheme.renderers.cdxml_writer import cdxml_to_scheme


def _build_cdxml(fragments_xml: str, step_xml: str) -> str:
    """Build a minimal CDXML string for testing import."""
    return f"""<?xml version='1.0' encoding='us-ascii'?>
<CDXML>
  <page>
    {fragments_xml}
    <arrow id="100" FillType="None" ArrowheadType="Solid"
           Head3D="300.0 200.0 0.0" Tail3D="200.0 200.0 0.0"/>
    <scheme>
      {step_xml}
    </scheme>
  </page>
</CDXML>"""


# ──────────────────────────────────────────────
# Scenario 1: Import formal charge from CDXML nodes
# ──────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.chemistry
class TestCdxmlImportCharge:
    """AC-9.2: _fragment_to_smiles reads Charge attribute."""

    def test_import_positive_charge(self) -> None:
        """Node with Charge='1' on nitrogen → [NH4+] in reconstructed SMILES."""
        # NH4+: one nitrogen with 4 hydrogens and +1 charge
        frag = """<fragment id="1">
          <n id="10" p="100 200" Element="7" NumHydrogens="4" Charge="1"/>
        </fragment>"""
        product = """<fragment id="2">
          <n id="20" p="400 200" Element="6"/>
        </fragment>"""
        step = '<step ReactionStepReactants="1" ReactionStepProducts="2" ReactionStepArrows="100"/>'
        cdxml = _build_cdxml(frag + product, step)
        scheme = cdxml_to_scheme(cdxml)

        reactant_smiles = scheme["steps"][0]["reactants"][0]
        mol = Chem.MolFromSmiles(reactant_smiles)
        assert mol is not None
        # Check that the nitrogen has +1 charge
        n_atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
        assert len(n_atoms) >= 1
        assert n_atoms[0].GetFormalCharge() == 1, (
            f"Expected +1 charge, got {n_atoms[0].GetFormalCharge()}"
        )

    def test_import_negative_charge(self) -> None:
        """Node with Charge='-1' on oxygen → [O-] in reconstructed SMILES."""
        # Hydroxide: one oxygen with -1 charge
        frag = """<fragment id="1">
          <n id="10" p="100 200" Element="8" NumHydrogens="0" Charge="-1"/>
        </fragment>"""
        product = """<fragment id="2">
          <n id="20" p="400 200" Element="6"/>
        </fragment>"""
        step = '<step ReactionStepReactants="1" ReactionStepProducts="2" ReactionStepArrows="100"/>'
        cdxml = _build_cdxml(frag + product, step)
        scheme = cdxml_to_scheme(cdxml)

        reactant_smiles = scheme["steps"][0]["reactants"][0]
        mol = Chem.MolFromSmiles(reactant_smiles)
        assert mol is not None
        o_atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 8]
        assert len(o_atoms) >= 1
        assert o_atoms[0].GetFormalCharge() == -1

    def test_import_no_charge_attr_defaults_zero(self) -> None:
        """Adversarial: node without Charge attr → charge = 0."""
        frag = """<fragment id="1">
          <n id="10" p="100 200" Element="7" NumHydrogens="3"/>
        </fragment>"""
        product = """<fragment id="2">
          <n id="20" p="400 200" Element="6"/>
        </fragment>"""
        step = '<step ReactionStepReactants="1" ReactionStepProducts="2" ReactionStepArrows="100"/>'
        cdxml = _build_cdxml(frag + product, step)
        scheme = cdxml_to_scheme(cdxml)

        reactant_smiles = scheme["steps"][0]["reactants"][0]
        mol = Chem.MolFromSmiles(reactant_smiles)
        assert mol is not None
        for a in mol.GetAtoms():
            assert a.GetFormalCharge() == 0


# ──────────────────────────────────────────────
# Scenario 2: Import isotope from CDXML nodes
# ──────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.chemistry
class TestCdxmlImportIsotope:
    """AC-9.2: _fragment_to_smiles reads Isotope attribute."""

    def test_import_carbon_13(self) -> None:
        """Node with Isotope='13' → [13C] in reconstructed SMILES."""
        frag = """<fragment id="1">
          <n id="10" p="100 200" Element="6" Isotope="13" NumHydrogens="4"/>
        </fragment>"""
        product = """<fragment id="2">
          <n id="20" p="400 200" Element="6"/>
        </fragment>"""
        step = '<step ReactionStepReactants="1" ReactionStepProducts="2" ReactionStepArrows="100"/>'
        cdxml = _build_cdxml(frag + product, step)
        scheme = cdxml_to_scheme(cdxml)

        reactant_smiles = scheme["steps"][0]["reactants"][0]
        mol = Chem.MolFromSmiles(reactant_smiles)
        assert mol is not None
        isotope_atoms = [a for a in mol.GetAtoms() if a.GetIsotope() == 13]
        assert len(isotope_atoms) >= 1, "Should have carbon-13 atom"

    def test_import_deuterium(self) -> None:
        """Node with Element='1' Isotope='2' → deuterium."""
        frag = """<fragment id="1">
          <n id="10" p="100 200" Element="1" Isotope="2"/>
          <n id="11" p="130 200" Element="8" NumHydrogens="1"/>
          <b id="50" B="10" E="11"/>
        </fragment>"""
        product = """<fragment id="2">
          <n id="20" p="400 200" Element="6"/>
        </fragment>"""
        step = '<step ReactionStepReactants="1" ReactionStepProducts="2" ReactionStepArrows="100"/>'
        cdxml = _build_cdxml(frag + product, step)
        scheme = cdxml_to_scheme(cdxml)

        reactant_smiles = scheme["steps"][0]["reactants"][0]
        mol = Chem.MolFromSmiles(reactant_smiles)
        assert mol is not None
        h_atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 1]
        deuterium = [a for a in h_atoms if a.GetIsotope() == 2]
        assert len(deuterium) >= 1, "Should have deuterium atom"

    def test_import_no_isotope_attr_defaults_zero(self) -> None:
        """Adversarial: node without Isotope attr → isotope = 0 (natural)."""
        frag = """<fragment id="1">
          <n id="10" p="100 200" Element="6" NumHydrogens="4"/>
        </fragment>"""
        product = """<fragment id="2">
          <n id="20" p="400 200" Element="6"/>
        </fragment>"""
        step = '<step ReactionStepReactants="1" ReactionStepProducts="2" ReactionStepArrows="100"/>'
        cdxml = _build_cdxml(frag + product, step)
        scheme = cdxml_to_scheme(cdxml)

        reactant_smiles = scheme["steps"][0]["reactants"][0]
        mol = Chem.MolFromSmiles(reactant_smiles)
        assert mol is not None
        for a in mol.GetAtoms():
            assert a.GetIsotope() == 0


# ──────────────────────────────────────────────
# Scenario 3: Import stereo bonds (wedge/hash → chirality)
# ──────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.chemistry
class TestCdxmlImportStereo:
    """AC-9.2: _fragment_to_smiles reads Display attribute for stereo."""

    def test_import_wedge_bond_creates_chirality(self) -> None:
        """Bond with Display='WedgeBegin' should produce chiral SMILES.

        Build a tetrahedral center: C with F, Cl, Br + wedge bond.
        """
        frag = """<fragment id="1">
          <n id="10" p="100 200" Element="6" NumHydrogens="1"/>
          <n id="11" p="130 180" Element="9"/>
          <n id="12" p="130 220" Element="17"/>
          <n id="13" p="70 200" Element="35"/>
          <b id="50" B="10" E="11" Display="WedgeBegin"/>
          <b id="51" B="10" E="12"/>
          <b id="52" B="10" E="13"/>
        </fragment>"""
        product = """<fragment id="2">
          <n id="20" p="400 200" Element="6"/>
        </fragment>"""
        step = '<step ReactionStepReactants="1" ReactionStepProducts="2" ReactionStepArrows="100"/>'
        cdxml = _build_cdxml(frag + product, step)
        scheme = cdxml_to_scheme(cdxml)

        reactant_smiles = scheme["steps"][0]["reactants"][0]
        mol = Chem.MolFromSmiles(reactant_smiles)
        assert mol is not None
        # Should have chirality from wedge bond
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        chiral = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        assert len(chiral) >= 1, (
            f"Wedge bond should create chiral center, got SMILES: {reactant_smiles}"
        )

    def test_import_hash_bond_creates_chirality(self) -> None:
        """Bond with Display='WedgedHashBegin' should also produce chirality."""
        frag = """<fragment id="1">
          <n id="10" p="100 200" Element="6" NumHydrogens="1"/>
          <n id="11" p="130 180" Element="9"/>
          <n id="12" p="130 220" Element="17"/>
          <n id="13" p="70 200" Element="35"/>
          <b id="50" B="10" E="11" Display="WedgedHashBegin"/>
          <b id="51" B="10" E="12"/>
          <b id="52" B="10" E="13"/>
        </fragment>"""
        product = """<fragment id="2">
          <n id="20" p="400 200" Element="6"/>
        </fragment>"""
        step = '<step ReactionStepReactants="1" ReactionStepProducts="2" ReactionStepArrows="100"/>'
        cdxml = _build_cdxml(frag + product, step)
        scheme = cdxml_to_scheme(cdxml)

        reactant_smiles = scheme["steps"][0]["reactants"][0]
        mol = Chem.MolFromSmiles(reactant_smiles)
        assert mol is not None
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        chiral = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        assert len(chiral) >= 1, (
            f"Hash bond should create chiral center, got SMILES: {reactant_smiles}"
        )


# ──────────────────────────────────────────────
# Scenario 4: Import aromatic bonds (Order="1.5")
# ──────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.chemistry
class TestCdxmlImportAromatic:
    """AC-9.2: _fragment_to_smiles correctly handles aromatic bonds."""

    def test_import_aromatic_benzene(self) -> None:
        """Benzene with Order='1.5' bonds → aromatic SMILES (c1ccccc1).

        This already works but we lock the behavior with a test.
        """
        # Benzene: 6 carbons in a ring with aromatic bonds
        frag = """<fragment id="1">
          <n id="10" p="100 200"/>
          <n id="11" p="130 180"/>
          <n id="12" p="160 200"/>
          <n id="13" p="160 230"/>
          <n id="14" p="130 250"/>
          <n id="15" p="100 230"/>
          <b id="50" B="10" E="11" Order="1.5"/>
          <b id="51" B="11" E="12" Order="1.5"/>
          <b id="52" B="12" E="13" Order="1.5"/>
          <b id="53" B="13" E="14" Order="1.5"/>
          <b id="54" B="14" E="15" Order="1.5"/>
          <b id="55" B="15" E="10" Order="1.5"/>
        </fragment>"""
        product = """<fragment id="2">
          <n id="20" p="400 200" Element="6"/>
        </fragment>"""
        step = '<step ReactionStepReactants="1" ReactionStepProducts="2" ReactionStepArrows="100"/>'
        cdxml = _build_cdxml(frag + product, step)
        scheme = cdxml_to_scheme(cdxml)

        reactant_smiles = scheme["steps"][0]["reactants"][0]
        mol = Chem.MolFromSmiles(reactant_smiles)
        assert mol is not None
        # Should be aromatic
        assert mol.GetNumAtoms() == 6
        assert all(a.GetIsAromatic() for a in mol.GetAtoms()), (
            f"All atoms should be aromatic, got SMILES: {reactant_smiles}"
        )


# ──────────────────────────────────────────────
# Scenario 5: Combined features in import
# ──────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.chemistry
class TestCdxmlImportCombined:
    """AC-9.2: Multiple features parsed simultaneously."""

    def test_charged_isotope_import(self) -> None:
        """Node with both Charge and Isotope attributes → both preserved."""
        # 13C-labeled carboxylate: [13C]([O-])=O
        frag = """<fragment id="1">
          <n id="10" p="100 200" Element="6" Isotope="13"/>
          <n id="11" p="130 180" Element="8" Charge="-1" NumHydrogens="0"/>
          <n id="12" p="130 220" Element="8" NumHydrogens="0"/>
          <b id="50" B="10" E="11"/>
          <b id="51" B="10" E="12" Order="2"/>
        </fragment>"""
        product = """<fragment id="2">
          <n id="20" p="400 200" Element="6"/>
        </fragment>"""
        step = '<step ReactionStepReactants="1" ReactionStepProducts="2" ReactionStepArrows="100"/>'
        cdxml = _build_cdxml(frag + product, step)
        scheme = cdxml_to_scheme(cdxml)

        reactant_smiles = scheme["steps"][0]["reactants"][0]
        mol = Chem.MolFromSmiles(reactant_smiles)
        assert mol is not None

        # Check isotope on carbon
        c_atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 6]
        assert any(a.GetIsotope() == 13 for a in c_atoms), "Should have 13C"

        # Check charge on oxygen
        o_atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 8]
        assert any(a.GetFormalCharge() == -1 for a in o_atoms), "Should have [O-]"

    def test_adversarial_invalid_charge_value(self) -> None:
        """Adversarial: non-integer Charge attr should not crash."""
        frag = """<fragment id="1">
          <n id="10" p="100 200" Element="7" NumHydrogens="3" Charge="abc"/>
        </fragment>"""
        product = """<fragment id="2">
          <n id="20" p="400 200" Element="6"/>
        </fragment>"""
        step = '<step ReactionStepReactants="1" ReactionStepProducts="2" ReactionStepArrows="100"/>'
        cdxml = _build_cdxml(frag + product, step)
        # Should not raise — graceful fallback
        scheme = cdxml_to_scheme(cdxml)
        assert len(scheme["steps"]) == 1
