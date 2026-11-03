"""Phase 19 CLI tests — chemscheme lookup + suggest commands.

AC-19.4: CLI lookup <query> and suggest <reaction-type>
AC-19.6: Adversarial (empty query, unknown reaction, --json)
"""

import json

from click.testing import CliRunner

from chemscheme.cli import main

# ---------------------------------------------------------------------------
# AC-19.4: chemscheme lookup
# ---------------------------------------------------------------------------


class TestCLILookup:
    """CLI lookup command."""

    def test_lookup_nbs(self) -> None:
        """GIVEN 'NBS', WHEN chemscheme lookup NBS, THEN shows result."""
        runner = CliRunner()
        result = runner.invoke(main, ["lookup", "NBS"])
        assert result.exit_code == 0
        assert "NBS" in result.output

    def test_lookup_json(self) -> None:
        """GIVEN --json, WHEN lookup, THEN outputs valid JSON."""
        runner = CliRunner()
        result = runner.invoke(main, ["lookup", "NBS", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "results" in data
        assert len(data["results"]) >= 1

    def test_lookup_category_filter(self) -> None:
        """GIVEN --category catalyst, WHEN lookup Pd, THEN only catalysts."""
        runner = CliRunner()
        result = runner.invoke(main, ["lookup", "Pd", "--category", "catalyst", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        for r in data["results"]:
            assert r["category"] == "catalyst"

    def test_lookup_no_match(self) -> None:
        """GIVEN unknown query, WHEN lookup, THEN helpful message."""
        runner = CliRunner()
        result = runner.invoke(main, ["lookup", "xyznonexistent999"])
        assert result.exit_code == 0
        assert "No" in result.output or "no" in result.output

    def test_lookup_empty_query(self) -> None:
        """AC-19.6: GIVEN no query, WHEN lookup, THEN error or usage."""
        runner = CliRunner()
        result = runner.invoke(main, ["lookup"])
        # Click should complain about missing argument
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# AC-19.4: chemscheme suggest
# ---------------------------------------------------------------------------


class TestCLISuggest:
    """CLI suggest command."""

    def test_suggest_suzuki(self) -> None:
        """GIVEN 'Suzuki coupling', WHEN suggest, THEN shows conditions."""
        runner = CliRunner()
        result = runner.invoke(main, ["suggest", "Suzuki coupling"])
        assert result.exit_code == 0
        assert "Pd" in result.output or "catalyst" in result.output.lower()

    def test_suggest_json(self) -> None:
        """GIVEN --json, WHEN suggest, THEN outputs valid JSON."""
        runner = CliRunner()
        result = runner.invoke(main, ["suggest", "Suzuki coupling", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["status"] == "ok"
        assert "catalyst" in data

    def test_suggest_unknown(self) -> None:
        """AC-19.6: GIVEN unknown reaction, WHEN suggest, THEN helpful error."""
        runner = CliRunner()
        result = runner.invoke(main, ["suggest", "xyznonexistent"])
        assert result.exit_code == 0
        assert "not found" in result.output.lower() or "available" in result.output.lower()

    def test_suggest_partial_match(self) -> None:
        """AC-19.6: GIVEN 'stille', WHEN suggest, THEN matches Stille coupling."""
        runner = CliRunner()
        result = runner.invoke(main, ["suggest", "stille", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["status"] == "ok"
