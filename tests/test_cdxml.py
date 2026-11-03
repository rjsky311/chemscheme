"""Tests for CDXML export, import, and round-trip.

Phase 7 — AC-7.4, AC-7.5, AC-7.6.
BDD-first: all tests RED before implementation.
"""

import xml.etree.ElementTree as ET

import pytest


@pytest.mark.unit
class TestCdxmlExport:
    """AC-7.4: scheme.json → .cdxml export."""

    def test_export_single_step_produces_valid_xml(self, sample_scheme: dict) -> None:
        """Single-step scheme exports to well-formed XML."""
        from chemscheme.renderers.cdxml_writer import scheme_to_cdxml

        cdxml = scheme_to_cdxml(sample_scheme)
        assert isinstance(cdxml, str)
        # Must be parseable XML
        root = ET.fromstring(cdxml)
        assert root.tag == "CDXML"

    def test_export_has_page_element(self, sample_scheme: dict) -> None:
        """CDXML contains a page element."""
        from chemscheme.renderers.cdxml_writer import scheme_to_cdxml

        cdxml = scheme_to_cdxml(sample_scheme)
        root = ET.fromstring(cdxml)
        pages = root.findall("page")
        assert len(pages) >= 1

    def test_export_has_fragment_per_molecule(self, sample_scheme: dict) -> None:
        """Each reactant/product has a fragment element."""
        from chemscheme.renderers.cdxml_writer import scheme_to_cdxml

        cdxml = scheme_to_cdxml(sample_scheme)
        root = ET.fromstring(cdxml)
        page = root.find("page")
        assert page is not None
        fragments = page.findall("fragment")
        # 1-step scheme: 2 reactants + 1 product = 3 fragments
        assert len(fragments) >= 3

    def test_export_fragment_has_nodes_and_bonds(self, sample_scheme: dict) -> None:
        """Each fragment contains node (n) and bond (b) elements."""
        from chemscheme.renderers.cdxml_writer import scheme_to_cdxml

        cdxml = scheme_to_cdxml(sample_scheme)
        root = ET.fromstring(cdxml)
        page = root.find("page")
        assert page is not None
        fragment = page.findall("fragment")[0]
        nodes = fragment.findall("n")
        bonds = fragment.findall("b")
        assert len(nodes) > 0
        assert len(bonds) > 0

    def test_export_has_reaction_arrow(self, sample_scheme: dict) -> None:
        """CDXML contains an arrow element for the reaction."""
        from chemscheme.renderers.cdxml_writer import scheme_to_cdxml

        cdxml = scheme_to_cdxml(sample_scheme)
        root = ET.fromstring(cdxml)
        page = root.find("page")
        assert page is not None
        arrows = page.findall("arrow")
        assert len(arrows) >= 1

    def test_export_has_scheme_and_step(self, sample_scheme: dict) -> None:
        """CDXML contains scheme/step elements linking reactants → products."""
        from chemscheme.renderers.cdxml_writer import scheme_to_cdxml

        cdxml = scheme_to_cdxml(sample_scheme)
        root = ET.fromstring(cdxml)
        page = root.find("page")
        assert page is not None
        scheme_elem = page.find("scheme")
        assert scheme_elem is not None
        step_elem = scheme_elem.find("step")
        assert step_elem is not None
        # Step should reference reactants, products, and arrow
        assert step_elem.get("ReactionStepReactants") is not None
        assert step_elem.get("ReactionStepProducts") is not None
        assert step_elem.get("ReactionStepArrows") is not None

    def test_export_arrow_above_as_text(self, sample_scheme: dict) -> None:
        """arrow_above conditions appear as text near the arrow."""
        from chemscheme.renderers.cdxml_writer import scheme_to_cdxml

        cdxml = scheme_to_cdxml(sample_scheme)
        # The conditions text should appear somewhere in the CDXML
        assert "Pd(PPh3)4" in cdxml or "Pd" in cdxml

    def test_export_multistep_scheme(self, sample_scheme_3step: dict) -> None:
        """3-step scheme produces 3 arrows and multiple fragments."""
        from chemscheme.renderers.cdxml_writer import scheme_to_cdxml

        cdxml = scheme_to_cdxml(sample_scheme_3step)
        root = ET.fromstring(cdxml)
        page = root.find("page")
        assert page is not None
        arrows = page.findall("arrow")
        assert len(arrows) == 3
        scheme_elem = page.find("scheme")
        assert scheme_elem is not None
        steps = scheme_elem.findall("step")
        assert len(steps) == 3

    def test_export_node_positions_exist(self, sample_scheme: dict) -> None:
        """Nodes have 2D position attributes (p="x y")."""
        from chemscheme.renderers.cdxml_writer import scheme_to_cdxml

        cdxml = scheme_to_cdxml(sample_scheme)
        root = ET.fromstring(cdxml)
        page = root.find("page")
        assert page is not None
        fragment = page.findall("fragment")[0]
        node = fragment.findall("n")[0]
        assert node.get("p") is not None
        # Position should contain two space-separated numbers
        parts = node.get("p").split()
        assert len(parts) == 2


@pytest.mark.unit
class TestCdxmlImport:
    """AC-7.5: .cdxml → scheme.json import."""

    def test_import_basic_cdxml(self, sample_scheme: dict) -> None:
        """Import a CDXML string and get a valid scheme dict."""
        from chemscheme.renderers.cdxml_writer import cdxml_to_scheme, scheme_to_cdxml

        cdxml = scheme_to_cdxml(sample_scheme)
        result = cdxml_to_scheme(cdxml)
        assert isinstance(result, dict)
        assert "schema_version" in result
        assert "steps" in result
        assert len(result["steps"]) > 0

    def test_import_step_has_required_fields(self, sample_scheme: dict) -> None:
        """Imported steps have id, reactants, products, arrow_above."""
        from chemscheme.renderers.cdxml_writer import cdxml_to_scheme, scheme_to_cdxml

        cdxml = scheme_to_cdxml(sample_scheme)
        result = cdxml_to_scheme(cdxml)
        step = result["steps"][0]
        assert "id" in step
        assert "reactants" in step
        assert "products" in step
        assert "arrow_above" in step

    def test_import_reactants_are_valid_smiles(self, sample_scheme: dict) -> None:
        """Imported reactant SMILES are valid (RDKit-parseable)."""
        from rdkit import Chem

        from chemscheme.renderers.cdxml_writer import cdxml_to_scheme, scheme_to_cdxml

        cdxml = scheme_to_cdxml(sample_scheme)
        result = cdxml_to_scheme(cdxml)
        for smi in result["steps"][0]["reactants"]:
            mol = Chem.MolFromSmiles(smi)
            assert mol is not None, f"Invalid SMILES in import: {smi}"

    def test_import_multistep(self, sample_scheme_3step: dict) -> None:
        """3-step CDXML imports back as 3 steps."""
        from chemscheme.renderers.cdxml_writer import cdxml_to_scheme, scheme_to_cdxml

        cdxml = scheme_to_cdxml(sample_scheme_3step)
        result = cdxml_to_scheme(cdxml)
        assert len(result["steps"]) == 3


@pytest.mark.unit
class TestCdxmlRoundTrip:
    """AC-7.6: Round-trip scheme.json → CDXML → scheme.json preserves data."""

    def test_round_trip_step_count(self, sample_scheme: dict) -> None:
        """Step count preserved through round-trip."""
        from chemscheme.renderers.cdxml_writer import cdxml_to_scheme, scheme_to_cdxml

        cdxml = scheme_to_cdxml(sample_scheme)
        result = cdxml_to_scheme(cdxml)
        assert len(result["steps"]) == len(sample_scheme["steps"])

    def test_round_trip_reactant_smiles(self, sample_scheme: dict) -> None:
        """Reactant SMILES preserved (canonical form) through round-trip."""
        from rdkit import Chem

        from chemscheme.renderers.cdxml_writer import cdxml_to_scheme, scheme_to_cdxml

        cdxml = scheme_to_cdxml(sample_scheme)
        result = cdxml_to_scheme(cdxml)

        # Compare canonical SMILES sets (order may differ after canonicalization)
        orig = sample_scheme["steps"][0]["reactants"]
        res = result["steps"][0]["reactants"]
        orig_canon = {Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in orig}
        res_canon = {Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in res}
        assert orig_canon == res_canon

    def test_round_trip_product_smiles(self, sample_scheme: dict) -> None:
        """Product SMILES preserved through round-trip."""
        from rdkit import Chem

        from chemscheme.renderers.cdxml_writer import cdxml_to_scheme, scheme_to_cdxml

        cdxml = scheme_to_cdxml(sample_scheme)
        result = cdxml_to_scheme(cdxml)

        # Compare canonical SMILES sets (order may differ after canonicalization)
        orig = sample_scheme["steps"][0]["products"]
        res = result["steps"][0]["products"]
        orig_canon = {Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in orig}
        res_canon = {Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in res}
        assert orig_canon == res_canon

    def test_round_trip_arrow_above(self, sample_scheme: dict) -> None:
        """arrow_above text preserved through round-trip."""
        from chemscheme.renderers.cdxml_writer import cdxml_to_scheme, scheme_to_cdxml

        cdxml = scheme_to_cdxml(sample_scheme)
        result = cdxml_to_scheme(cdxml)
        assert result["steps"][0]["arrow_above"] == sample_scheme["steps"][0]["arrow_above"]

    def test_round_trip_arrow_below(self, sample_scheme: dict) -> None:
        """arrow_below text preserved through round-trip."""
        from chemscheme.renderers.cdxml_writer import cdxml_to_scheme, scheme_to_cdxml

        cdxml = scheme_to_cdxml(sample_scheme)
        result = cdxml_to_scheme(cdxml)
        assert result["steps"][0].get("arrow_below") == sample_scheme["steps"][0].get("arrow_below")

    def test_round_trip_3step_all_steps(self, sample_scheme_3step: dict) -> None:
        """All 3 steps preserved with correct reactants/products through round-trip."""

        from chemscheme.renderers.cdxml_writer import cdxml_to_scheme, scheme_to_cdxml

        cdxml = scheme_to_cdxml(sample_scheme_3step)
        result = cdxml_to_scheme(cdxml)

        assert len(result["steps"]) == 3
        for i in range(3):
            orig_step = sample_scheme_3step["steps"][i]
            result_step = result["steps"][i]
            # Same number of reactants and products
            assert len(result_step["reactants"]) == len(orig_step["reactants"])
            assert len(result_step["products"]) == len(orig_step["products"])
            # arrow_above preserved
            assert result_step["arrow_above"] == orig_step["arrow_above"]

    def test_round_trip_yield_preserved(self, sample_scheme: dict) -> None:
        """Optional yield field preserved through round-trip."""
        from chemscheme.renderers.cdxml_writer import cdxml_to_scheme, scheme_to_cdxml

        cdxml = scheme_to_cdxml(sample_scheme)
        result = cdxml_to_scheme(cdxml)
        assert result["steps"][0].get("yield") == sample_scheme["steps"][0].get("yield")

    def test_round_trip_schema_version(self, sample_scheme: dict) -> None:
        """schema_version preserved through round-trip."""
        from chemscheme.renderers.cdxml_writer import cdxml_to_scheme, scheme_to_cdxml

        cdxml = scheme_to_cdxml(sample_scheme)
        result = cdxml_to_scheme(cdxml)
        assert result["schema_version"] == sample_scheme["schema_version"]

    def test_round_trip_empty_consumes_step_ids(self) -> None:
        """Empty consumes_step_ids=[] should survive round-trip (C1 regression)."""
        from chemscheme.renderers.cdxml_writer import cdxml_to_scheme, scheme_to_cdxml

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["c1ccccc1"],
                    "products": ["c1ccc(Br)cc1"],
                    "arrow_above": "NBS",
                    "consumes_step_ids": [],
                },
            ],
        }
        cdxml = scheme_to_cdxml(scheme)
        result = cdxml_to_scheme(cdxml)
        assert result["steps"][0]["consumes_step_ids"] == []
