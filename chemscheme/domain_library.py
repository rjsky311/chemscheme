"""Domain knowledge library — reagent lookup + reaction pattern suggestions.

Provides searchable access to:
- Reagent abbreviations with categories (reagent_aliases.json v3)
- Named reaction patterns with typical conditions (reaction_patterns.json)
"""

from __future__ import annotations

import importlib.resources
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Fallback paths for source-tree development
_LEGACY_DATA_DIR = Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReagentInfo:
    """A reagent entry from the domain library."""

    abbreviation: str
    name: str
    category: str = "reagent"
    smiles: str | None = None


@dataclass(frozen=True)
class ReactionPattern:
    """A named reaction with typical conditions."""

    name: str
    reagents: tuple[str, ...] = ()
    catalyst: str = ""
    solvents: tuple[str, ...] = ()
    bases: tuple[str, ...] = ()
    key_reagent: str = ""
    temperature: str = ""
    time: str = ""
    atmosphere: str | None = None
    description: str = ""


@dataclass(frozen=True)
class ReactionMatch:
    """Result of reaction type identification with confidence scoring.

    Attributes:
        name: Matched reaction name (e.g., "Suzuki coupling") or "" if no match.
        confidence: Score from 0.0 (no match) to 1.0 (high confidence).
        pattern: The matched ReactionPattern, or None if no match.
    """

    name: str
    confidence: float
    pattern: ReactionPattern | None = None

    def __str__(self) -> str:
        """Return reaction name for backward-compatible string usage."""
        return self.name


# ---------------------------------------------------------------------------
# Data loading (importlib.resources with legacy fallback)
# ---------------------------------------------------------------------------

_reagent_cache: list[ReagentInfo] | None = None
_reaction_cache: list[ReactionPattern] | None = None


def _invalidate_all_caches() -> None:
    """Invalidate all module-level caches (A6/G4 fix).

    Called after user library mutations to ensure both domain_library
    and text_extraction pick up changes in long-running processes.
    """
    global _reagent_cache, _reaction_cache  # noqa: PLW0603
    _reagent_cache = None
    _reaction_cache = None
    # Also invalidate text_extraction caches if loaded
    try:
        from chemscheme import text_extraction as _te

        _te._solvent_names = None  # noqa: SLF001
        _te._reagent_abbrevs = None  # noqa: SLF001
    except (ImportError, AttributeError):
        pass


def _load_data_text(filename: str) -> str:
    """Load a data file from chemscheme.data package."""
    try:
        ref = importlib.resources.files("chemscheme.data").joinpath(filename)
        return ref.read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, TypeError):
        logger.debug("importlib.resources failed for %s, using legacy path", filename)

    legacy = _LEGACY_DATA_DIR / filename
    if legacy.exists():
        return legacy.read_text(encoding="utf-8")

    msg = f"{filename} not found in chemscheme/data/"
    raise FileNotFoundError(msg)


def _load_reagents() -> list[ReagentInfo]:
    """Load and cache reagent entries from reagent_aliases.json."""
    global _reagent_cache  # noqa: PLW0603
    if _reagent_cache is not None:
        return _reagent_cache

    text = _load_data_text("reagent_aliases.json")
    data = json.loads(text)
    abbrevs = data.get("abbreviations", {})

    entries: list[ReagentInfo] = []
    for abbr, info in abbrevs.items():
        if abbr.startswith("_"):  # Skip metadata keys like _comment_* (C1 fix)
            continue
        if isinstance(info, dict):
            entries.append(
                ReagentInfo(
                    abbreviation=abbr,
                    name=info.get("name", abbr),
                    category=info.get("category", "reagent"),
                    smiles=info.get("smiles"),
                )
            )
        else:
            # Legacy format: abbr -> name string
            entries.append(ReagentInfo(abbreviation=abbr, name=str(info)))

    # Merge user-defined reagents (user overrides built-in on conflict)
    user_entries = _load_user_reagents()
    if user_entries:
        user_abbrs = {r.abbreviation for r in user_entries}
        merged = [r for r in entries if r.abbreviation not in user_abbrs]
        merged.extend(user_entries)
        entries = merged

    _reagent_cache = entries
    return entries


def _load_reactions() -> list[ReactionPattern]:
    """Load and cache reaction patterns from reaction_patterns.json."""
    global _reaction_cache  # noqa: PLW0603
    if _reaction_cache is not None:
        return _reaction_cache

    text = _load_data_text("reaction_patterns.json")
    data = json.loads(text)
    # G5 fix: guard against "reactions": null in JSON
    reactions_raw = data.get("reactions") or []

    patterns: list[ReactionPattern] = []
    for i, r in enumerate(reactions_raw):
        if not isinstance(r, dict):
            logger.warning("Skipping non-dict reaction entry at index %d", i)
            continue
        # Validate list fields — guard against string-to-chars bug
        # G4 fix: explicit null → empty list (JSON null passes .get default)
        reagents = r.get("reagents") or []
        solvents = r.get("solvents") or []
        bases = r.get("bases") or []
        if not isinstance(reagents, list):
            logger.warning("reaction[%d] reagents is not a list, coercing", i)
            reagents = [str(reagents)]
        if not isinstance(solvents, list):
            logger.warning("reaction[%d] solvents is not a list, coercing", i)
            solvents = [str(solvents)]
        if not isinstance(bases, list):
            logger.warning("reaction[%d] bases is not a list, coercing", i)
            bases = [str(bases)]
        patterns.append(
            ReactionPattern(
                name=r.get("name", ""),
                reagents=tuple(reagents),
                catalyst=r.get("catalyst", ""),
                solvents=tuple(solvents),
                bases=tuple(bases),
                key_reagent=r.get("key_reagent", ""),
                temperature=r.get("temperature", ""),
                time=r.get("time", ""),
                atmosphere=r.get("atmosphere"),
                description=r.get("description", ""),
            )
        )

    _reaction_cache = patterns
    return patterns


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def lookup_reagent(query: str, *, category: str | None = None) -> list[ReagentInfo]:
    """Search reagents by abbreviation, name, or category.

    Args:
        query: Search string (case-insensitive, partial match).
        category: Optional category filter (catalyst, base, solvent, etc.).

    Returns:
        List of matching ReagentInfo, sorted by relevance (exact match first).
    """
    if not query:
        return []

    q = query.lower()
    reagents = _load_reagents()

    results: list[tuple[int, ReagentInfo]] = []
    for r in reagents:
        # Category filter
        if category and r.category != category:
            continue

        # Scoring: lower = better match
        abbr_lower = r.abbreviation.lower()
        name_lower = r.name.lower()

        if abbr_lower == q:
            results.append((0, r))  # exact abbreviation match
        elif q in abbr_lower:
            results.append((1, r))  # partial abbreviation match
        elif q in name_lower:
            results.append((2, r))  # name contains query
        # No match → skip

    results.sort(key=lambda x: x[0])
    return [r for _, r in results]


def lookup_reaction(query: str) -> list[ReactionPattern]:
    """Search reaction patterns by name.

    Args:
        query: Search string (case-insensitive, partial match).

    Returns:
        List of matching ReactionPattern.
    """
    if not query:
        return []

    q = query.lower()
    reactions = _load_reactions()

    results: list[tuple[int, ReactionPattern]] = []
    for rp in reactions:
        name_lower = rp.name.lower()
        if name_lower == q:
            results.append((0, rp))
        elif q in name_lower:
            results.append((1, rp))

    results.sort(key=lambda x: x[0])
    return [rp for _, rp in results]


def suggest_conditions(reaction_type: str) -> ReactionPattern | None:
    """Suggest typical conditions for a named reaction.

    Returns the best match: exact match preferred, then unique partial match.
    If multiple partial matches exist, returns the first one (by definition order).

    Args:
        reaction_type: Reaction name (case-insensitive, partial match).

    Returns:
        Best matching ReactionPattern, or None if not found.
    """
    matches = lookup_reaction(reaction_type)
    if not matches:
        return None
    # Prefer exact match (score 0 from lookup_reaction)
    q = reaction_type.lower()
    for m in matches:
        if m.name.lower() == q:
            return m
    # Single partial match → unambiguous
    return matches[0]


def identify_reaction_type(condition_text: str) -> ReactionMatch:
    """Identify a named reaction type from condition text.

    Uses unified weighted scoring across all dimensions (key_reagent, catalyst,
    base, solvent, atmosphere) to find the best matching reaction pattern.

    Weights:
        key_reagent hit: +3.0 | key_reagent absent: -5.0 | catalyst: +2.0
        base: +1.5 | atmosphere: +1.0 | solvent: +0.5

    Args:
        condition_text: Combined condition text (e.g., "Pd(PPh3)4, K2CO3 in THF").

    Returns:
        ReactionMatch with name, confidence (0.0-1.0), and matched pattern.
    """
    no_match = ReactionMatch("", 0.0)

    if not condition_text or not condition_text.strip():
        return no_match

    combined_lower = condition_text.lower()
    patterns = _load_reactions()

    # Score ALL patterns in a single pass across all dimensions
    scored: list[tuple[ReactionPattern, float, int]] = []  # (pattern, score, dim_hits)

    for pat in patterns:
        score = 0.0
        dimension_hits = 0
        kr_hit = False  # R2 fix: hoist for use in evidence threshold

        # --- Dimension 1: key_reagent (weight +3.0 / -5.0) ---
        if pat.key_reagent:
            kr = pat.key_reagent
            kr_hit = False
            if len(kr) <= 3:
                # Short symbols: case-sensitive + uppercase-left + lowercase-right boundary
                # C5 fix: prevents "Ligand" matching "Li", "TMSN3" matching "Sn"
                # G1 fix: (?<![A-Z]) prevents "NBS" matching "B" key_reagent
                # R2 note: lowercase left allowed for chemistry notation
                # (ArSnBu3 = aryl + Sn, PhMgBr = phenyl + Mg)
                if re.search(rf"(?<![A-Z]){re.escape(kr)}(?![a-z])", condition_text):
                    kr_hit = True
            elif kr.lower() in combined_lower:
                kr_hit = True

            if kr_hit:
                score += 3.0
                dimension_hits += 1
            else:
                # Mandatory key_reagent absent → heavy penalty
                # Prevents Stille matching on catalyst alone without Sn
                score -= 5.0

        # --- Dimension 2: catalyst (weight +2.0) ---
        if pat.catalyst:
            # C-N2 fix: split on "/" for multi-variant catalysts
            cat_variants = [c.strip().lower() for c in pat.catalyst.split("/")]
            if any(cv in combined_lower for cv in cat_variants):
                score += 2.0
                dimension_hits += 1

        # --- Dimension 3: base (weight +1.5) ---
        if pat.bases and any(b.lower() in combined_lower for b in pat.bases):
            score += 1.5
            dimension_hits += 1

        # --- Dimension 4: solvent (weight +0.5) ---
        if pat.solvents and any(s.lower() in combined_lower for s in pat.solvents):
            score += 0.5
            dimension_hits += 1

        # --- Dimension 5: atmosphere (weight +1.0) ---
        if pat.atmosphere and pat.atmosphere.lower() in combined_lower:
            score += 1.0
            dimension_hits += 1

        # Must match at least 1 dimension with positive total
        # C4 fix: require ≥2 dimension hits OR key_reagent hit to avoid
        # spurious matches on solvent-only or single-base-only inputs
        # R2 fix: use actual kr_hit (not inferred from score)
        if score > 0 and (dimension_hits >= 2 or kr_hit):
            scored.append((pat, score, dimension_hits))

    if not scored:
        return no_match

    # Sort by score descending, then by name for stable tiebreak
    scored.sort(key=lambda x: (-x[1], x[0].name))
    best_pat, best_score, _ = scored[0]

    # Confidence: normalize to 0-1 range
    # Max theoretical = 3.0 + 2.0 + 1.5 + 0.5 + 1.0 = 8.0
    max_score = 8.0
    confidence = min(best_score / max_score, 1.0)

    # Ambiguity: if top-2 scores are within 0.5, reduce confidence
    if len(scored) >= 2:
        gap = scored[0][1] - scored[1][1]
        if gap < 0.5:
            confidence *= 0.7

    return ReactionMatch(best_pat.name, round(confidence, 3), best_pat)


# ---------------------------------------------------------------------------
# User-extensible library (~/.chemscheme/)
# ---------------------------------------------------------------------------

_USER_REAGENTS_FILE = "custom_reagents.json"
_USER_REACTIONS_FILE = "custom_reactions.json"


def _get_user_data_dir() -> Path:
    """Get or create user data directory ~/.chemscheme/."""
    d = Path.home() / ".chemscheme"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_user_reagents_raw(user_dir: Path | None = None) -> dict:
    """Load raw user reagent data from custom_reagents.json."""
    if user_dir is None:
        user_dir = _get_user_data_dir()
    fp = user_dir / _USER_REAGENTS_FILE
    if not fp.exists():
        return {"abbreviations": {}}
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            logger.warning("Invalid %s (not a dict), treating as empty", fp)
            return {"abbreviations": {}}
        # A5 fix: validate abbreviations is a dict, not list or other type
        abbrevs = data.get("abbreviations")
        if abbrevs is not None and not isinstance(abbrevs, dict):
            logger.warning("Invalid abbreviations in %s, treating as empty", fp)
            data["abbreviations"] = {}
        return data
    except (json.JSONDecodeError, OSError):
        # G2 fix: log at error level (this is data loss risk)
        logger.error("Corrupt %s — treating as empty. Check file manually.", fp)
        return {"abbreviations": {}}


def _save_user_reagents_raw(data: dict, user_dir: Path | None = None) -> None:
    """Save user reagent data to custom_reagents.json (atomic write)."""
    import tempfile as _tf

    if user_dir is None:
        user_dir = _get_user_data_dir()
    fp = user_dir / _USER_REAGENTS_FILE
    # G1/A5 fix: atomic write via temp file + replace to prevent corruption
    content = json.dumps(data, indent=2, ensure_ascii=False)
    fd, tmp_path = _tf.mkstemp(dir=str(user_dir), suffix=".tmp")
    try:
        with open(fd, "w", encoding="utf-8") as f:
            f.write(content)
        Path(tmp_path).replace(fp)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def _load_user_reagents() -> list[ReagentInfo]:
    """Load user-defined reagents from ~/.chemscheme/custom_reagents.json."""
    data = _load_user_reagents_raw()
    # G3 fix: guard against "abbreviations": null in JSON
    abbrevs = data.get("abbreviations") or {}
    entries: list[ReagentInfo] = []
    for abbr, info in abbrevs.items():
        if not isinstance(info, dict):
            continue
        # R2-03 fix: validate/coerce inner fields
        name = info.get("name", abbr)
        category = info.get("category", "reagent")
        smiles = info.get("smiles")
        if not isinstance(name, str):
            logger.warning("Skipping custom reagent %r: invalid name type", abbr)
            continue
        if not isinstance(category, str):
            category = "reagent"
        if smiles is not None and not isinstance(smiles, str):
            smiles = None
        entries.append(
            ReagentInfo(
                abbreviation=abbr,
                name=name,
                category=category,
                smiles=smiles,
            )
        )
    return entries


def list_user_reagents() -> list[ReagentInfo]:
    """List all user-defined custom reagents."""
    return _load_user_reagents()


def add_user_reagent(
    abbreviation: str,
    name: str,
    category: str,
    smiles: str | None = None,
) -> ReagentInfo:
    """Add a custom reagent to user library.

    Args:
        abbreviation: Short name (e.g., "MyNFA").
        name: Full name.
        category: Category (catalyst, solvent, reagent, etc.).
        smiles: Optional SMILES string (validated via RDKit).

    Returns:
        The created ReagentInfo.

    Raises:
        ValueError: If abbreviation is empty or SMILES is invalid.
    """
    if not abbreviation or not abbreviation.strip():
        msg = "Abbreviation cannot be empty"
        raise ValueError(msg)

    canonical_smiles = None
    if smiles:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            msg = f"Invalid SMILES: {smiles!r}"
            raise ValueError(msg)
        canonical_smiles = Chem.MolToSmiles(mol)

    user_dir = _get_user_data_dir()
    data = _load_user_reagents_raw(user_dir)
    abbrevs = data.setdefault("abbreviations", {})
    entry: dict[str, str] = {"name": name, "category": category}
    if canonical_smiles:
        entry["smiles"] = canonical_smiles
    abbrevs[abbreviation] = entry
    _save_user_reagents_raw(data, user_dir)

    # A6/G4 fix: centralized cache invalidation
    _invalidate_all_caches()

    return ReagentInfo(
        abbreviation=abbreviation,
        name=name,
        category=category,
        smiles=canonical_smiles,
    )


def remove_user_reagent(abbreviation: str) -> bool:
    """Remove a custom reagent from user library.

    Returns True if removed, False if not found in user library.
    Built-in reagents cannot be removed.
    """
    user_dir = _get_user_data_dir()
    data = _load_user_reagents_raw(user_dir)
    abbrevs = data.get("abbreviations", {})
    if abbreviation not in abbrevs:
        return False
    del abbrevs[abbreviation]
    _save_user_reagents_raw(data, user_dir)

    # A6/G4 fix: centralized cache invalidation
    _invalidate_all_caches()
    return True
