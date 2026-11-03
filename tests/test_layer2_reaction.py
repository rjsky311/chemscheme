"""Tests for Layer 2 reaction validation (AC-6b.1, AC-6b.2, AC-6b.3)."""

import pytest


# ---------------------------------------------------------------------------
# AC-6b.1  ReactionPredictor Protocol + MockPredictor
# ---------------------------------------------------------------------------
class TestReactionPredictorProtocol:
    """AC-6b.1: Protocol definition and MockPredictor implementation."""

    @pytest.mark.unit
    def test_mock_predictor_exists(self) -> None:
        """MockPredictor class is importable."""
        from chemscheme.validators.layer2_reaction import MockPredictor

        predictor = MockPredictor()
        assert predictor is not None

    @pytest.mark.unit
    def test_mock_predictor_conforms_to_protocol(self) -> None:
        """MockPredictor has predict_products method."""
        from chemscheme.validators.layer2_reaction import MockPredictor

        predictor = MockPredictor()
        assert hasattr(predictor, "predict_products")
        assert callable(predictor.predict_products)

    @pytest.mark.unit
    def test_mock_predictor_returns_result(self) -> None:
        """MockPredictor.predict_products returns a PredictionResult."""
        from chemscheme.validators.layer2_reaction import MockPredictor, PredictionResult

        predictor = MockPredictor()
        result = predictor.predict_products(
            reactants=["c1ccc(Br)cc1", "OB(O)c1ccccc1"],
            reagents="Pd(PPh3)4, K2CO3",
        )
        assert isinstance(result, PredictionResult)
        assert result.feasible is True
        assert result.confidence >= 0.0


# ---------------------------------------------------------------------------
# AC-6b.2  IBM RXN / ASKCOS call (mocked)
# ---------------------------------------------------------------------------
class TestReactionValidation:
    """AC-6b.2: Reaction validation with mocked predictor."""

    @pytest.mark.integration
    def test_validate_reaction_with_mock(self) -> None:
        """validate_reaction uses predictor and returns result."""
        from chemscheme.validators.layer2_reaction import (
            MockPredictor,
            validate_reaction,
        )

        result = validate_reaction(
            reactants=["c1ccc(Br)cc1", "OB(O)c1ccccc1"],
            products=["c1ccc(-c2ccccc2)cc1"],
            reagents="Pd(PPh3)4, K2CO3",
            predictor=MockPredictor(),
        )
        assert result.layer == "reaction"
        assert result.errors == ()

    @pytest.mark.integration
    def test_validate_reaction_feasibility_check(self) -> None:
        """Reaction flagged as infeasible produces warning."""
        from chemscheme.validators.layer2_reaction import (
            MockPredictor,
            PredictionResult,
            validate_reaction,
        )

        class InfeasiblePredictor(MockPredictor):
            def predict_products(self, reactants: list[str], reagents: str) -> PredictionResult:
                return PredictionResult(
                    feasible=False,
                    confidence=0.3,
                    predicted_products=[],
                    source="mock",
                )

        result = validate_reaction(
            reactants=["CC", "OO"],
            products=["CCO"],
            reagents="nothing",
            predictor=InfeasiblePredictor(),
        )
        assert len(result.warnings) > 0


# ---------------------------------------------------------------------------
# AC-6b.3  API failure -> graceful fallback
# ---------------------------------------------------------------------------
class TestReactionFallback:
    """AC-6b.3: Predictor failure does not block workflow."""

    @pytest.mark.unit
    def test_predictor_exception_graceful(self) -> None:
        """Predictor raising exception -> warning, not error."""
        from chemscheme.validators.layer2_reaction import (
            MockPredictor,
            PredictionResult,
            validate_reaction,
        )

        class BrokenPredictor(MockPredictor):
            def predict_products(self, reactants: list[str], reagents: str) -> PredictionResult:
                raise ConnectionError("API down")

        result = validate_reaction(
            reactants=["c1ccccc1"],
            products=["c1ccc(O)cc1"],
            reagents="H2O2",
            predictor=BrokenPredictor(),
        )
        # Should NOT raise, should produce warning
        assert result.errors == ()
        assert len(result.warnings) > 0
        assert any("unavailable" in w.lower() or "failed" in w.lower() for w in result.warnings)

    @pytest.mark.unit
    def test_no_predictor_skips_layer(self) -> None:
        """No predictor provided -> layer skipped with info message."""
        from chemscheme.validators.layer2_reaction import validate_reaction

        result = validate_reaction(
            reactants=["c1ccccc1"],
            products=["c1ccc(O)cc1"],
            reagents="H2O2",
            predictor=None,
        )
        assert result.errors == ()
        assert any("skipped" in w.lower() for w in result.warnings)
