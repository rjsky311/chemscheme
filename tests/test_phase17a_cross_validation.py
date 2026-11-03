"""Phase 17a: Cross-Validation Infrastructure tests.

BDD-first RED tests — all must FAIL before implementation.
"""

from __future__ import annotations

import time

import pytest

from chemscheme.models import AgreementLevel, RecognitionResult
from chemscheme.recognition.cross_validator import (
    cross_validate,
    inchikey_agree,
    smiles_agree,
    text_token_contain,
)
from chemscheme.recognition.mocks import (
    MockDiagramParser,
    MockMoleculeRecognizer,
    MockTextExtractor,
)
from chemscheme.recognition.protocols import (
    DiagramParser,
    MoleculeRecognizer,
    RecognizerProtocol,
    TextExtractor,
)
from chemscheme.recognition.runner import run_tool

# ===========================================================================
# AC-17a.1: RecognitionResult data type
# ===========================================================================


class TestRecognitionResult:
    """RecognitionResult frozen dataclass with agreement scoring."""

    def test_create_basic(self) -> None:
        r = RecognitionResult(
            value="CCO",
            confidence=0.95,
            source="test",
            agreement=AgreementLevel.HIGH,
            sources=("molscribe", "decimer"),
        )
        assert r.value == "CCO"
        assert r.confidence == 0.95
        assert r.source == "test"
        assert r.agreement == AgreementLevel.HIGH
        assert r.sources == ("molscribe", "decimer")
        assert r.disagreements == ()
        assert r.needs_human_review is False

    def test_frozen_immutable(self) -> None:
        r = RecognitionResult(
            value="CCO",
            confidence=0.9,
            source="test",
            agreement=AgreementLevel.HIGH,
            sources=("a",),
        )
        with pytest.raises(AttributeError):
            r.value = "CCCO"  # type: ignore[misc]

    def test_list_to_tuple_coercion(self) -> None:
        """Lists provided for tuple fields are coerced to tuples."""
        r = RecognitionResult(
            value="CCO",
            confidence=0.9,
            source="test",
            agreement=AgreementLevel.HIGH,
            sources=["a", "b"],  # type: ignore[arg-type]
            disagreements=["c"],  # type: ignore[arg-type]
        )
        assert isinstance(r.sources, tuple)
        assert isinstance(r.disagreements, tuple)

    def test_needs_human_review_default_false(self) -> None:
        r = RecognitionResult(
            value="CCO",
            confidence=0.9,
            source="test",
            agreement=AgreementLevel.HIGH,
            sources=("a",),
        )
        assert r.needs_human_review is False

    def test_needs_human_review_explicit(self) -> None:
        r = RecognitionResult(
            value="",
            confidence=0.2,
            source="test",
            agreement=AgreementLevel.LOW,
            sources=("a", "b"),
            needs_human_review=True,
        )
        assert r.needs_human_review is True

    def test_agreement_level_enum_values(self) -> None:
        assert AgreementLevel.HIGH.value == "high"
        assert AgreementLevel.MEDIUM.value == "medium"
        assert AgreementLevel.LOW.value == "low"
        assert AgreementLevel.SINGLE.value == "single"


# ===========================================================================
# AC-17a.2: RecognizerProtocol interfaces
# ===========================================================================


class TestProtocols:
    """Protocol interfaces are runtime-checkable."""

    def test_molecule_recognizer_protocol(self) -> None:
        mock = MockMoleculeRecognizer(value="CCO", confidence=0.9)
        assert isinstance(mock, MoleculeRecognizer)
        assert isinstance(mock, RecognizerProtocol)

    def test_diagram_parser_protocol(self) -> None:
        mock = MockDiagramParser(steps=1)
        assert isinstance(mock, DiagramParser)
        assert isinstance(mock, RecognizerProtocol)

    def test_text_extractor_protocol(self) -> None:
        mock = MockTextExtractor(text="K2CO3")
        assert isinstance(mock, TextExtractor)
        assert isinstance(mock, RecognizerProtocol)

    def test_non_conforming_object_fails(self) -> None:
        """An object without recognize() does not satisfy Protocol."""

        class NotARecognizer:
            pass

        assert not isinstance(NotARecognizer(), RecognizerProtocol)


# ===========================================================================
# AC-17a.3: CrossValidator generic framework
# ===========================================================================


class TestCrossValidator:
    """cross_validate merges results with agreement scoring."""

    def test_three_agree_high(self) -> None:
        results = [
            RecognitionResult(
                value="c1ccccc1",
                confidence=0.9,
                source="a",
                agreement=AgreementLevel.SINGLE,
                sources=("a",),
            ),
            RecognitionResult(
                value="c1ccccc1",
                confidence=0.85,
                source="b",
                agreement=AgreementLevel.SINGLE,
                sources=("b",),
            ),
            RecognitionResult(
                value="c1ccccc1",
                confidence=0.88,
                source="c",
                agreement=AgreementLevel.SINGLE,
                sources=("c",),
            ),
        ]
        merged = cross_validate(results, smiles_agree)
        assert merged.agreement == AgreementLevel.HIGH
        assert merged.confidence == 0.9  # max
        assert merged.needs_human_review is False
        assert len(merged.sources) == 3

    def test_two_of_three_agree_medium(self) -> None:
        results = [
            RecognitionResult(
                value="c1ccccc1",
                confidence=0.9,
                source="a",
                agreement=AgreementLevel.SINGLE,
                sources=("a",),
            ),
            RecognitionResult(
                value="c1ccccc1",
                confidence=0.85,
                source="b",
                agreement=AgreementLevel.SINGLE,
                sources=("b",),
            ),
            RecognitionResult(
                value="CCO",
                confidence=0.7,
                source="c",
                agreement=AgreementLevel.SINGLE,
                sources=("c",),
            ),
        ]
        merged = cross_validate(results, smiles_agree)
        assert merged.agreement == AgreementLevel.MEDIUM
        assert "c" in merged.disagreements or "CCO" in str(merged.disagreements)

    def test_all_disagree_low(self) -> None:
        results = [
            RecognitionResult(
                value="CCO",
                confidence=0.8,
                source="a",
                agreement=AgreementLevel.SINGLE,
                sources=("a",),
            ),
            RecognitionResult(
                value="CCCO",
                confidence=0.7,
                source="b",
                agreement=AgreementLevel.SINGLE,
                sources=("b",),
            ),
            RecognitionResult(
                value="CC(O)C",
                confidence=0.6,
                source="c",
                agreement=AgreementLevel.SINGLE,
                sources=("c",),
            ),
        ]
        merged = cross_validate(results, smiles_agree)
        assert merged.agreement == AgreementLevel.LOW
        assert merged.needs_human_review is True

    def test_single_result(self) -> None:
        results = [
            RecognitionResult(
                value="CCO",
                confidence=0.8,
                source="a",
                agreement=AgreementLevel.SINGLE,
                sources=("a",),
            ),
        ]
        merged = cross_validate(results, smiles_agree)
        assert merged.agreement == AgreementLevel.SINGLE
        assert merged.confidence == 0.8

    def test_two_results_agree(self) -> None:
        results = [
            RecognitionResult(
                value="CCO",
                confidence=0.9,
                source="a",
                agreement=AgreementLevel.SINGLE,
                sources=("a",),
            ),
            RecognitionResult(
                value="CCO",
                confidence=0.85,
                source="b",
                agreement=AgreementLevel.SINGLE,
                sources=("b",),
            ),
        ]
        merged = cross_validate(results, smiles_agree)
        assert merged.agreement == AgreementLevel.HIGH
        assert merged.confidence == 0.9

    def test_two_results_disagree(self) -> None:
        results = [
            RecognitionResult(
                value="CCO",
                confidence=0.9,
                source="a",
                agreement=AgreementLevel.SINGLE,
                sources=("a",),
            ),
            RecognitionResult(
                value="CCCO",
                confidence=0.85,
                source="b",
                agreement=AgreementLevel.SINGLE,
                sources=("b",),
            ),
        ]
        merged = cross_validate(results, smiles_agree)
        assert merged.agreement == AgreementLevel.LOW
        assert merged.needs_human_review is True

    def test_empty_results_raises(self) -> None:
        with pytest.raises(ValueError, match="[Ee]mpty|[Nn]o results"):
            cross_validate([], smiles_agree)


# ===========================================================================
# AC-17a.4: Agreement functions
# ===========================================================================


class TestAgreementFunctions:
    """Domain-specific agreement functions."""

    # --- smiles_agree ---
    def test_smiles_agree_identical(self) -> None:
        assert smiles_agree("CCO", "CCO") is True

    def test_smiles_agree_canonical_equivalent(self) -> None:
        """Different representations of same molecule agree."""
        assert smiles_agree("c1ccccc1", "C1=CC=CC=C1") is True

    def test_smiles_agree_different(self) -> None:
        assert smiles_agree("CCO", "CCCO") is False

    def test_smiles_agree_invalid_first(self) -> None:
        """Invalid SMILES does not crash, returns False."""
        assert smiles_agree("not_a_smiles", "CCO") is False

    def test_smiles_agree_invalid_both(self) -> None:
        assert smiles_agree("xxx", "yyy") is False

    def test_smiles_agree_empty_strings(self) -> None:
        assert smiles_agree("", "") is False

    def test_smiles_agree_whitespace(self) -> None:
        assert smiles_agree(" CCO ", "CCO") is True

    # --- inchikey_agree ---
    def test_inchikey_agree_same(self) -> None:
        assert (
            inchikey_agree(
                "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
                "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
            )
            is True
        )

    def test_inchikey_agree_connectivity_layer(self) -> None:
        """First 14 chars (connectivity) match, rest differs."""
        assert (
            inchikey_agree(
                "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
                "LFQSCWFLJHTTHZ-DIFFERENT--X",
            )
            is True
        )

    def test_inchikey_agree_different(self) -> None:
        assert (
            inchikey_agree(
                "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
                "XXXXXXXXXYYYYY-UHFFFAOYSA-N",
            )
            is False
        )

    def test_inchikey_agree_empty(self) -> None:
        assert inchikey_agree("", "") is False

    # --- text_token_contain ---
    def test_text_token_contain_all_present(self) -> None:
        assert text_token_contain("K2CO3 DMF 80°C", "K2CO3, DMF, 80°C, 12h") is True

    def test_text_token_contain_missing_token(self) -> None:
        """OCR has token not in AI output → returns False."""
        assert text_token_contain("K2CO3 DMF N2", "K2CO3, DMF") is False

    def test_text_token_contain_empty_ocr(self) -> None:
        """Empty OCR text → trivially True (no tokens to check)."""
        assert text_token_contain("", "K2CO3, DMF") is True

    def test_text_token_contain_empty_ai(self) -> None:
        """Non-empty OCR, empty AI → False."""
        assert text_token_contain("K2CO3", "") is False

    def test_text_token_contain_case_insensitive(self) -> None:
        assert text_token_contain("k2co3", "K2CO3 in DMF") is True


# ===========================================================================
# AC-17a.5: Subprocess tool runner
# ===========================================================================


class TestToolRunner:
    """run_tool subprocess wrapper with timeout and error handling."""

    def test_successful_command(self) -> None:
        output = run_tool(["python", "-c", "print('hello')"])
        assert output.returncode == 0
        assert b"hello" in output.stdout
        assert output.elapsed >= 0

    def test_command_with_stderr(self) -> None:
        output = run_tool(["python", "-c", "import sys; sys.stderr.write('warn')"])
        assert output.returncode == 0
        assert "warn" in output.stderr

    def test_nonzero_exit(self) -> None:
        output = run_tool(["python", "-c", "raise SystemExit(1)"])
        assert output.returncode == 1

    def test_timeout_kills_process(self) -> None:
        output = run_tool(
            ["python", "-c", "import time; time.sleep(60)"],
            timeout=2,
        )
        assert output.returncode != 0
        assert "timeout" in output.stderr.lower() or output.returncode != 0

    def test_command_not_found(self) -> None:
        output = run_tool(["nonexistent_binary_xyz_123"])
        assert output.returncode != 0

    def test_input_data_piped(self) -> None:
        output = run_tool(
            ["python", "-c", "import sys; print(sys.stdin.read().strip())"],
            input_data=b"test_input",
        )
        assert output.returncode == 0
        assert b"test_input" in output.stdout

    def test_tool_output_fields(self) -> None:
        output = run_tool(["python", "-c", "print('ok')"])
        assert hasattr(output, "stdout")
        assert hasattr(output, "stderr")
        assert hasattr(output, "returncode")
        assert hasattr(output, "elapsed")
        assert isinstance(output.stdout, bytes)
        assert isinstance(output.stderr, str)
        assert isinstance(output.returncode, int)
        assert isinstance(output.elapsed, float)


# ===========================================================================
# AC-17a.6: Mock recognizers
# ===========================================================================


class TestMockRecognizers:
    """Configurable mock recognizers for testing."""

    def test_mock_molecule_recognizer(self) -> None:
        mock = MockMoleculeRecognizer(value="CCO", confidence=0.95)
        result = mock.recognize(b"fake_image")
        assert result.value == "CCO"
        assert result.confidence == 0.95
        assert result.source == "mock_molecule"

    def test_mock_molecule_recognizer_custom_source(self) -> None:
        mock = MockMoleculeRecognizer(value="CCO", confidence=0.9, source="custom")
        result = mock.recognize(b"fake_image")
        assert result.source == "custom"

    def test_mock_diagram_parser(self) -> None:
        mock = MockDiagramParser(steps=3)
        result = mock.recognize(b"fake_image")
        assert result.value  # non-empty
        assert result.confidence > 0

    def test_mock_text_extractor(self) -> None:
        mock = MockTextExtractor(text="K2CO3, DMF, 80°C")
        result = mock.recognize(b"fake_image")
        assert "K2CO3" in result.value

    def test_mock_with_delay(self) -> None:
        mock = MockMoleculeRecognizer(value="CCO", confidence=0.9, delay=0.1)
        t0 = time.monotonic()
        mock.recognize(b"fake_image")
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.09  # allow small timing tolerance


# ===========================================================================
# Integration: cross_validate with mock recognizers
# ===========================================================================


class TestCrossValidatorIntegration:
    """End-to-end: mock recognizers → cross_validate."""

    def test_three_mocks_agree(self) -> None:
        mocks = [
            MockMoleculeRecognizer(value="c1ccccc1", confidence=0.9),
            MockMoleculeRecognizer(value="c1ccccc1", confidence=0.85),
            MockMoleculeRecognizer(value="c1ccccc1", confidence=0.88),
        ]
        results = [m.recognize(b"img") for m in mocks]
        merged = cross_validate(results, smiles_agree)
        assert merged.agreement == AgreementLevel.HIGH

    def test_two_mocks_disagree_one(self) -> None:
        mocks = [
            MockMoleculeRecognizer(value="CCO", confidence=0.9),
            MockMoleculeRecognizer(value="CCO", confidence=0.85),
            MockMoleculeRecognizer(value="CCCO", confidence=0.7),
        ]
        results = [m.recognize(b"img") for m in mocks]
        merged = cross_validate(results, smiles_agree)
        assert merged.agreement == AgreementLevel.MEDIUM
