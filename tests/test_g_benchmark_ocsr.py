"""S68 Part 4 — 4-tool OCSR benchmark framework.

Closes S67 Part 5b backlog: Mathpix accuracy assumption was unverified
(99% marketing for general OCR; not chemistry-specific; not in MARCUS comparison;
no independent benchmark). Now that Phase 17c MARCUS-spec parity is achieved
(Part 3 added MolNexTR), the 4-tool benchmark is unblocked.

This module provides:
- canonical SMILES match scoring (reuses smiles_agree from cross_validator)
- per-tool result aggregation
- a `chemscheme benchmark-ocsr` CLI command for production use

Real-recognizer execution requires Docker daemon + 3 docker images built.
Tests use mocked recognizers — measures the framework, not real accuracy.
Production users: `docker build -t chemscheme-{molscribe,decimer,molnextr}
docker/{molscribe,decimer,molnextr}/`, set CHEMSCHEME_OCSR_DOCKER=1, then
`chemscheme benchmark-ocsr --input-dir tests/fixtures/ocsr_benchmark/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from chemscheme.cli import main as cli_main
from chemscheme.models import AgreementLevel, RecognitionResult


def _result(smiles: str, conf: float, source: str) -> RecognitionResult:
    return RecognitionResult(
        value=smiles,
        confidence=conf,
        source=source,
        agreement=AgreementLevel.SINGLE,
        sources=(source,),
    )


class _StubRec:
    def __init__(self, smiles: str, conf: float = 0.9, source: str = "stub") -> None:
        self._smiles = smiles
        self._conf = conf
        self._source = source

    def recognize(self, image: bytes, image_format: str = "png") -> RecognitionResult:
        return _result(self._smiles, self._conf, self._source)


# ──────────────────────────── Scoring primitives ────────────────────────────


class TestScoreSmilesMatch:
    """score_smiles_match(predicted, ground_truth) — canonical match via RDKit."""

    def test_module_importable(self) -> None:
        from chemscheme.recognition.benchmark import score_smiles_match

        assert callable(score_smiles_match)

    def test_exact_match_returns_true(self) -> None:
        from chemscheme.recognition.benchmark import score_smiles_match

        assert score_smiles_match("c1ccccc1", "c1ccccc1") is True

    def test_kekule_vs_aromatic_match_returns_true(self) -> None:
        """C1=CC=CC=C1 (Kekule benzene) and c1ccccc1 (aromatic) are the same molecule."""
        from chemscheme.recognition.benchmark import score_smiles_match

        assert score_smiles_match("C1=CC=CC=C1", "c1ccccc1") is True

    def test_different_molecules_return_false(self) -> None:
        from chemscheme.recognition.benchmark import score_smiles_match

        assert score_smiles_match("c1ccccc1", "CCO") is False

    def test_invalid_predicted_returns_false(self) -> None:
        from chemscheme.recognition.benchmark import score_smiles_match

        assert score_smiles_match("C(C(C", "c1ccccc1") is False

    def test_empty_predicted_returns_false(self) -> None:
        from chemscheme.recognition.benchmark import score_smiles_match

        assert score_smiles_match("", "c1ccccc1") is False


# ──────────────────────────── BenchmarkResult dataclass ────────────────────────────


class TestBenchmarkResultDataclass:
    """Per-(tool, image) row in the benchmark output table."""

    def test_dataclass_importable(self) -> None:
        from chemscheme.recognition.benchmark import BenchmarkResult

        r = BenchmarkResult(
            tool="molscribe",
            image_id="benzene",
            ground_truth="c1ccccc1",
            predicted="c1ccccc1",
            confidence=0.85,
            match=True,
            elapsed_seconds=1.2,
            error=None,
        )
        assert r.tool == "molscribe"
        assert r.match is True
        assert r.error is None

    def test_dataclass_records_failure(self) -> None:
        from chemscheme.recognition.benchmark import BenchmarkResult

        r = BenchmarkResult(
            tool="decimer",
            image_id="hard_nfa",
            ground_truth="c1ccccc1",
            predicted="",
            confidence=0.0,
            match=False,
            elapsed_seconds=0.0,
            error="docker daemon not reachable",
        )
        assert r.match is False
        assert "docker" in (r.error or "")


# ──────────────────────────── run_benchmark ────────────────────────────


class TestRunBenchmark:
    """run_benchmark(images, ground_truth, recognizers) → list[BenchmarkResult]."""

    def test_runs_all_recognizers_on_all_images(self, tmp_path) -> None:
        from chemscheme.recognition.benchmark import run_benchmark

        # 2 images × 2 tools = 4 results expected
        img_a = tmp_path / "a.png"
        img_a.write_bytes(b"\x89PNG\r\n\x1a\nfake_a")
        img_b = tmp_path / "b.png"
        img_b.write_bytes(b"\x89PNG\r\n\x1a\nfake_b")

        ground_truth = {"a": "c1ccccc1", "b": "CCO"}
        recognizers = {
            "tool1": _StubRec("c1ccccc1", 0.9, "tool1"),
            "tool2": _StubRec("CCO", 0.85, "tool2"),
        }

        results = run_benchmark(
            images={"a": img_a, "b": img_b},
            ground_truth=ground_truth,
            recognizers=recognizers,
        )
        assert len(results) == 4
        # Each (tool, image) appears exactly once
        keys = {(r.tool, r.image_id) for r in results}
        assert keys == {("tool1", "a"), ("tool1", "b"), ("tool2", "a"), ("tool2", "b")}

    def test_match_correctly_reflects_canonical_equality(self, tmp_path) -> None:
        from chemscheme.recognition.benchmark import run_benchmark

        img = tmp_path / "x.png"
        img.write_bytes(b"\x89PNG")

        recs = {
            "correct": _StubRec("c1ccccc1", 0.9, "correct"),
            "wrong": _StubRec("CCO", 0.8, "wrong"),
        }
        results = run_benchmark(
            images={"x": img},
            ground_truth={"x": "c1ccccc1"},
            recognizers=recs,
        )
        match_by_tool = {r.tool: r.match for r in results}
        assert match_by_tool["correct"] is True
        assert match_by_tool["wrong"] is False

    def test_recognizer_exception_recorded_as_error_no_crash(self, tmp_path) -> None:
        """A failing recognizer (e.g., no docker) → BenchmarkResult with error field set."""
        from chemscheme.recognition.benchmark import run_benchmark

        class _Raising:
            def recognize(self, image: bytes, image_format: str = "png"):
                msg = "docker daemon not reachable"
                raise RuntimeError(msg)

        img = tmp_path / "x.png"
        img.write_bytes(b"\x89PNG")
        results = run_benchmark(
            images={"x": img},
            ground_truth={"x": "c1ccccc1"},
            recognizers={"broken": _Raising()},
        )
        assert len(results) == 1
        assert results[0].error is not None
        assert "docker" in results[0].error
        assert results[0].match is False
        assert results[0].confidence == 0.0

    def test_records_elapsed_seconds(self, tmp_path) -> None:
        from chemscheme.recognition.benchmark import run_benchmark

        img = tmp_path / "x.png"
        img.write_bytes(b"\x89PNG")
        results = run_benchmark(
            images={"x": img},
            ground_truth={"x": "c1ccccc1"},
            recognizers={"t": _StubRec("c1ccccc1")},
        )
        assert results[0].elapsed_seconds >= 0.0

    def test_image_without_ground_truth_skipped_with_warning(self, tmp_path) -> None:
        """An image not in ground_truth dict is silently skipped (no crash, no false positive)."""
        from chemscheme.recognition.benchmark import run_benchmark

        img = tmp_path / "unknown.png"
        img.write_bytes(b"\x89PNG")
        results = run_benchmark(
            images={"unknown": img},
            ground_truth={},  # no entry for "unknown"
            recognizers={"t": _StubRec("c1ccccc1")},
        )
        # No results for image without ground truth (can't compute match)
        assert len(results) == 0


# ──────────────────────────── generate_report ────────────────────────────


class TestGenerateReport:
    """generate_report(results) → per-tool aggregated metrics."""

    def test_per_tool_accuracy_count(self) -> None:
        from chemscheme.recognition.benchmark import BenchmarkResult, generate_report

        results = [
            BenchmarkResult("a", "1", "c1ccccc1", "c1ccccc1", 0.9, True, 1.0, None),
            BenchmarkResult("a", "2", "CCO", "CCO", 0.8, True, 1.0, None),
            BenchmarkResult("a", "3", "c1ccncc1", "CCC", 0.7, False, 1.0, None),
            BenchmarkResult("b", "1", "c1ccccc1", "c1ccccc1", 0.9, True, 1.0, None),
            BenchmarkResult("b", "2", "CCO", "CCC", 0.8, False, 1.0, None),
            BenchmarkResult("b", "3", "c1ccncc1", "c1ccncc1", 0.85, True, 1.0, None),
        ]
        report = generate_report(results)
        assert report["per_tool"]["a"]["accuracy"] == pytest.approx(2 / 3)
        assert report["per_tool"]["a"]["correct"] == 2
        assert report["per_tool"]["a"]["total"] == 3
        assert report["per_tool"]["b"]["accuracy"] == pytest.approx(2 / 3)

    def test_per_tool_average_confidence_only_on_correct(self) -> None:
        """Confidence calibration: avg confidence on CORRECT predictions vs INCORRECT.
        Distinguishes well-calibrated tool (high conf when right, low when wrong) from
        overconfident (high conf even when wrong)."""
        from chemscheme.recognition.benchmark import BenchmarkResult, generate_report

        results = [
            BenchmarkResult("a", "1", "c1ccccc1", "c1ccccc1", 0.9, True, 1.0, None),
            BenchmarkResult("a", "2", "CCO", "CCO", 0.85, True, 1.0, None),
            BenchmarkResult("a", "3", "c1ccncc1", "wrong", 0.95, False, 1.0, None),  # overconfident
        ]
        report = generate_report(results)
        a = report["per_tool"]["a"]
        assert a["avg_confidence_correct"] == pytest.approx((0.9 + 0.85) / 2)
        assert a["avg_confidence_incorrect"] == pytest.approx(0.95)
        # overconfident → calibration_warning surface in report
        assert a["avg_confidence_incorrect"] > a["avg_confidence_correct"]

    def test_excludes_errored_results_from_accuracy_denominator(self) -> None:
        """Tool failures (e.g., docker down) shouldn't count against accuracy
        (they count against COVERAGE — separate metric)."""
        from chemscheme.recognition.benchmark import BenchmarkResult, generate_report

        results = [
            BenchmarkResult("a", "1", "c1ccccc1", "c1ccccc1", 0.9, True, 1.0, None),
            BenchmarkResult("a", "2", "CCO", "", 0.0, False, 0.0, "docker down"),
            BenchmarkResult("a", "3", "c1ccncc1", "c1ccncc1", 0.85, True, 1.0, None),
        ]
        report = generate_report(results)
        a = report["per_tool"]["a"]
        # accuracy = 2/2 (only non-errored counted)
        assert a["accuracy"] == pytest.approx(1.0)
        assert a["coverage"] == pytest.approx(2 / 3)  # 2 of 3 attempted successfully
        assert a["errors"] == 1

    def test_overall_summary_includes_total_images_and_tools(self) -> None:
        from chemscheme.recognition.benchmark import BenchmarkResult, generate_report

        results = [
            BenchmarkResult("a", "1", "c1ccccc1", "c1ccccc1", 0.9, True, 1.0, None),
            BenchmarkResult("b", "1", "c1ccccc1", "c1ccccc1", 0.85, True, 1.0, None),
        ]
        report = generate_report(results)
        assert report["summary"]["total_images"] == 1
        assert report["summary"]["total_tools"] == 2
        assert report["summary"]["total_attempts"] == 2

    def test_empty_results_returns_empty_report_no_division_error(self) -> None:
        from chemscheme.recognition.benchmark import generate_report

        report = generate_report([])
        assert report["per_tool"] == {}
        assert report["summary"]["total_images"] == 0


# ──────────────────────────── CLI command ────────────────────────────


class TestBenchmarkOcsrCLI:
    """`chemscheme benchmark-ocsr` end-to-end."""

    def test_command_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli_main, ["benchmark-ocsr", "--help"])
        assert result.exit_code == 0
        assert "benchmark" in result.output.lower()

    def test_command_missing_inputs_exits_non_zero(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli_main, ["benchmark-ocsr"])
        assert result.exit_code != 0  # required arg missing

    def test_command_runs_with_mock_recognizers(self, tmp_path) -> None:
        """End-to-end: ground-truth JSON + image dir → JSON report.

        Uses mocked recognizers via MOCK env flag; real-tool gating is separate.
        """
        # Build minimal ground-truth + image dir
        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        (img_dir / "benzene.png").write_bytes(b"\x89PNG_benzene")
        (img_dir / "ethanol.png").write_bytes(b"\x89PNG_ethanol")

        gt = tmp_path / "ground_truth.json"
        gt.write_text(json.dumps({"benzene": "c1ccccc1", "ethanol": "CCO"}), encoding="utf-8")

        out = tmp_path / "report.json"

        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "benchmark-ocsr",
                "--input-dir",
                str(img_dir),
                "--ground-truth",
                str(gt),
                "--output",
                str(out),
                "--mock",  # use mock recognizers (don't require docker)
            ],
        )
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert out.exists()
        report = json.loads(out.read_text(encoding="utf-8"))
        assert "per_tool" in report
        assert "summary" in report
        assert report["summary"]["total_images"] == 2

    def test_command_handles_missing_ground_truth_keys_gracefully(self, tmp_path) -> None:
        """Image present but absent from ground-truth → skipped, command still 0."""
        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        (img_dir / "benzene.png").write_bytes(b"\x89PNG")
        (img_dir / "unlabeled.png").write_bytes(b"\x89PNG")

        gt = tmp_path / "gt.json"
        gt.write_text(json.dumps({"benzene": "c1ccccc1"}), encoding="utf-8")

        out = tmp_path / "report.json"
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "benchmark-ocsr",
                "--input-dir",
                str(img_dir),
                "--ground-truth",
                str(gt),
                "--output",
                str(out),
                "--mock",
            ],
        )
        assert result.exit_code == 0
        report = json.loads(out.read_text(encoding="utf-8"))
        assert report["summary"]["total_images"] == 1  # unlabeled skipped


# ──────────────────────────── Ground-truth fixture present ────────────────────────────


class TestBenchmarkFixturePresent:
    """A small starter ground-truth fixture must ship with the project so users
    can `chemscheme benchmark-ocsr --input-dir tests/fixtures/ocsr_benchmark/`
    out of the box. Production users grow this set over time."""

    def test_fixture_dir_exists(self) -> None:

        d = Path(__file__).parent / "fixtures" / "ocsr_benchmark"
        assert d.exists() and d.is_dir(), f"missing benchmark fixture dir: {d}"

    def test_ground_truth_json_exists_and_valid(self) -> None:

        gt_file = Path(__file__).parent / "fixtures" / "ocsr_benchmark" / "ground_truth.json"
        assert gt_file.exists(), f"missing ground_truth.json: {gt_file}"
        gt = json.loads(gt_file.read_text(encoding="utf-8"))
        assert isinstance(gt, dict) and len(gt) >= 3, (
            "ground_truth.json should contain ≥ 3 entries (small starter set)"
        )
        # Each value must be a valid SMILES (canonical via RDKit)
        from rdkit import Chem

        for name, smi in gt.items():
            assert isinstance(smi, str) and smi, (
                f"ground_truth['{name}'] must be a non-empty SMILES"
            )
            assert Chem.MolFromSmiles(smi) is not None, (
                f"ground_truth['{name}']={smi!r} is not a valid SMILES"
            )
