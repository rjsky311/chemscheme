"""Phase 0 smoke tests — verify core dependencies work on this platform.

# RED-EXEMPT: compatibility spike — these tests validate environment setup,
# not application behavior. They should PASS immediately if deps are installed.
"""

import pytest


@pytest.mark.smoke
class TestRDKitSmoke:
    """AC-0.2: RDKit installs and performs basic operations."""

    def test_rdkit_import(self) -> None:
        """RDKit can be imported."""
        from rdkit import Chem

        assert Chem is not None

    def test_rdkit_mol_from_smiles(self) -> None:
        """RDKit can parse a SMILES string into a molecule."""
        from rdkit import Chem

        mol = Chem.MolFromSmiles("c1ccccc1")
        assert mol is not None
        assert mol.GetNumAtoms() == 6

    def test_rdkit_canonical_smiles(self) -> None:
        """RDKit canonical SMILES is deterministic."""
        from rdkit import Chem

        # Two representations of benzene
        s1 = Chem.MolToSmiles(Chem.MolFromSmiles("c1ccccc1"))
        s2 = Chem.MolToSmiles(Chem.MolFromSmiles("C1=CC=CC=C1"))
        assert s1 == s2

    def test_rdkit_invalid_smiles(self) -> None:
        """RDKit returns None for invalid SMILES."""
        from rdkit import Chem

        mol = Chem.MolFromSmiles("not_a_smiles")
        assert mol is None


_has_indigo = True
try:
    import indigo as _indigo_check  # noqa: F401
except ImportError:
    _has_indigo = False


@pytest.mark.smoke
@pytest.mark.skipif(not _has_indigo, reason="indigo not installed")
class TestIndigoSmoke:
    """AC-0.2 extension: Indigo installs and renders on Python 3.14."""

    def test_indigo_import(self) -> None:
        """Indigo can be imported."""
        from indigo import Indigo

        assert Indigo is not None

    def test_indigo_load_molecule(self) -> None:
        """Indigo can load a molecule from SMILES."""
        from indigo import Indigo

        indigo = Indigo()
        mol = indigo.loadMolecule("c1ccccc1")
        assert mol.canonicalSmiles() == "c1ccccc1"

    def test_indigo_svg_render(self) -> None:
        """Indigo can render a molecule to SVG."""
        from indigo import Indigo
        from indigo.renderer import IndigoRenderer

        indigo = Indigo()
        renderer = IndigoRenderer(indigo)
        mol = indigo.loadMolecule("c1ccccc1")

        indigo.setOption("render-output-format", "svg")
        svg = renderer.renderToBuffer(mol)
        assert len(svg) > 0
        assert b"<svg" in svg


@pytest.mark.smoke
class TestClickSmoke:
    """CLI entrypoint smoke test."""

    def test_cli_help(self) -> None:
        """chemscheme --help runs without error."""
        from click.testing import CliRunner

        from chemscheme.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "publication-quality" in result.output

    def test_cli_version(self) -> None:
        """chemscheme --version outputs version string."""
        from click.testing import CliRunner

        from chemscheme.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output
