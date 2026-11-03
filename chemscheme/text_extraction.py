"""Text Extraction Pipeline — experimental text → structured reaction data.

Three-layer progressive extraction:
  L1: Condition regex (temperature, time, solvent, atmosphere, yield)
  L2: Chemical name finder (pattern matching + domain library)
  L3: Reaction role classifier (domain heuristics + Claude API fallback)
  L4: Text-to-scheme assembler

Zero new dependencies — reuses domain_library (Phase 19), name resolution
(Phase 5), and scheme_schema (Phase 1).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ExtractedConditions:
    """Reaction conditions extracted from text (L1)."""

    temperature: str | None = None
    time: str | None = None
    solvent: str | None = None
    atmosphere: str | None = None
    yield_pct: str | None = None


@dataclass
class FoundCompound:
    """A chemical compound identified in text (L2)."""

    name: str
    smiles: str = ""
    confidence: float = 0.0
    amount: str | None = None


@dataclass
class ClassifiedReaction:
    """A reaction with assigned roles (L3)."""

    reactants: list[FoundCompound] = field(default_factory=list)
    products: list[FoundCompound] = field(default_factory=list)
    reagents: list[FoundCompound] = field(default_factory=list)
    conditions: ExtractedConditions | None = None


# ---------------------------------------------------------------------------
# L1: Condition regex extractor (AC-20.1)
# ---------------------------------------------------------------------------

# Temperature patterns
_RE_TEMP_CELSIUS = re.compile(r"(-?\d+)\s*°\s*C")
_RE_TEMP_RT = re.compile(r"\b(room\s+temperature|rt|r\.t\.)\b", re.IGNORECASE)
_RE_TEMP_REFLUX = re.compile(r"\breflux\b", re.IGNORECASE)

# Time patterns
_RE_TIME = re.compile(
    r"(\d+(?:\.\d+)?)\s*(h(?:ours?)?|min(?:utes?)?|days?|s(?:econds?)?)\b",
    re.IGNORECASE,
)
_RE_TIME_OVERNIGHT = re.compile(r"\bovernight\b", re.IGNORECASE)

# Yield patterns
# Yield: require "yield" keyword context (G4+C1-R2 fix)
_RE_YIELD = re.compile(
    r"(?:"
    r"(\d+(?:\.\d+)?)\s*%\s*yield"  # "85% yield" or "85 % yield"
    r"|"
    r"yield[:\s]+(\d+(?:\.\d+)?)\s*%"  # "yield: 85%" or "yield 85%"
    r")",
    re.IGNORECASE,
)
_RE_YIELD_QUANT = re.compile(r"\bquantitative(?:\s+yield)?\b", re.IGNORECASE)

# Atmosphere patterns
_RE_ATMOS = re.compile(
    r"\b(?:under\s+(?:an?\s+)?)?(?:atmosphere\s+of\s+)?"
    r"(N[₂2]|(?:nitrogen)|Ar|(?:argon)|(?:air)|H[₂2]|(?:hydrogen)|O[₂2]|(?:oxygen))\b",
    re.IGNORECASE,
)

# Known solvents (loaded from domain library)
_solvent_names: set[str] | None = None


def _get_solvent_names() -> set[str]:
    """Load known solvent names from domain library (cached)."""
    global _solvent_names  # noqa: PLW0603
    if _solvent_names is not None:
        return _solvent_names
    try:
        from chemscheme.domain_library import _load_reagents

        reagents = _load_reagents()
        _solvent_names = {r.abbreviation for r in reagents if r.category == "solvent"}
        # Also add full names
        _solvent_names |= {r.name for r in reagents if r.category == "solvent"}
    except Exception:
        # Fallback: common solvents
        _solvent_names = {
            "THF",
            "DMF",
            "DCM",
            "DMSO",
            "MeCN",
            "MeOH",
            "EtOH",
            "toluene",
            "dioxane",
            "CHCl3",
            "hexane",
            "EtOAc",
            "Et2O",
            "DCE",
            "DME",
            "DMAc",
            "NMP",
            "iPrOH",
            "tBuOH",
            "xylene",
            "acetone",
            "water",
            "H2O",
        }
    return _solvent_names


def _normalize_atmosphere(raw: str) -> str:
    """Normalize atmosphere string to standard abbreviation."""
    low = raw.lower().strip()
    if low in ("n2", "n₂", "nitrogen"):
        return "N₂"
    if low in ("ar", "argon"):
        return "Ar"
    if low in ("h2", "h₂", "hydrogen"):
        return "H₂"
    if low in ("o2", "o₂", "oxygen"):
        return "O₂"
    if low == "air":
        return "air"
    return raw


def extract_conditions(text: str) -> ExtractedConditions:
    """Extract reaction conditions from text using regex patterns.

    Returns ExtractedConditions with all fields optional (None if not found).
    """
    if not text or not text.strip():
        return ExtractedConditions()

    result = ExtractedConditions()

    # Temperature
    m = _RE_TEMP_CELSIUS.search(text)
    if m:
        result.temperature = f"{m.group(1)} °C"
    elif _RE_TEMP_RT.search(text):
        result.temperature = "rt"
    elif _RE_TEMP_REFLUX.search(text):
        result.temperature = "reflux"

    # Time
    m = _RE_TIME.search(text)
    if m:
        result.time = f"{m.group(1)} {m.group(2)}"
        # Normalize: "hours" → "h", "minutes" → "min"
        result.time = re.sub(r"\bhours?\b", "h", result.time, flags=re.IGNORECASE)
        result.time = re.sub(r"\bminutes?\b", "min", result.time, flags=re.IGNORECASE)
        result.time = re.sub(r"\bdays?\b", "days", result.time, flags=re.IGNORECASE)
    elif _RE_TIME_OVERNIGHT.search(text):
        result.time = "overnight"

    # Solvent — case-insensitive, sorted for deterministic order, pick earliest (C7+NEW-2)
    solvents = _get_solvent_names()
    best_solvent: tuple[int, str] | None = None
    for name in sorted(solvents):
        m_sol = re.search(r"\b" + re.escape(name) + r"\b", text, re.IGNORECASE)
        if m_sol:
            pos = m_sol.start()
            if best_solvent is None or pos < best_solvent[0]:
                best_solvent = (pos, name)
    if best_solvent:
        result.solvent = best_solvent[1]

    # Atmosphere
    m = _RE_ATMOS.search(text)
    if m:
        result.atmosphere = _normalize_atmosphere(m.group(1))

    # Yield — require "yield" keyword context (G4+C1-R2 fix)
    if _RE_YIELD_QUANT.search(text):
        result.yield_pct = "quantitative"
    else:
        m = _RE_YIELD.search(text)
        if m:
            val = m.group(1) or m.group(2)
            if val:
                result.yield_pct = f"{val}%"

    return result


# ---------------------------------------------------------------------------
# L2: Chemical name finder (AC-20.2)
# ---------------------------------------------------------------------------

# Reagent abbreviation cache
_reagent_abbrevs: dict[str, dict[str, str]] | None = None


def _get_reagent_abbrevs() -> dict[str, dict[str, str]]:
    """Load reagent abbreviations from domain library (cached)."""
    global _reagent_abbrevs  # noqa: PLW0603
    if _reagent_abbrevs is not None:
        return _reagent_abbrevs
    try:
        from chemscheme.domain_library import _load_reagents

        reagents = _load_reagents()
        _reagent_abbrevs = {}
        for r in reagents:
            _reagent_abbrevs[r.abbreviation] = {
                "name": r.name,
                "smiles": r.smiles or "",
                "category": r.category,
            }
    except Exception:
        _reagent_abbrevs = {}
    return _reagent_abbrevs


# Pattern: numbered compound references like "compound 1", "(1)", "product 2"
_RE_COMPOUND_REF = re.compile(
    r"\b(compound|product|intermediate|substrate)\s+(\d+[a-z]?)\b", re.IGNORECASE
)

# Pattern: chemical formula (e.g., NaOH, KOH, Pd(PPh3)4) — module-level (NEW-5+G3)
_RE_FORMULA = re.compile(r"\b([A-Z][a-z]?(?:\d+)?(?:\([^)]+\)(?:\d+)?)?(?:[A-Z][a-z]?(?:\d+)?)*)\b")


def find_compounds(text: str) -> list[FoundCompound]:
    """Identify chemical compounds in text.

    Uses domain library matching + pattern-based detection.
    Returns list of FoundCompound with name, smiles, confidence.
    """
    if not text or not text.strip():
        return []

    found: list[FoundCompound] = []
    seen_names: set[str] = set()

    reagents = _get_reagent_abbrevs()

    # Match known reagent abbreviations from domain library
    for abbr, info in reagents.items():
        # Use word boundary or special chars for matching
        # Handle unicode subscripts and parentheses in names
        escaped = re.escape(abbr)
        # G6: set check before regex for short-circuit efficiency
        # C6 fix: case-insensitive matching for OCR-normalized text
        if abbr not in seen_names and re.search(
            r"(?<![a-zA-Z])" + escaped + r"(?![a-zA-Z])", text, re.IGNORECASE
        ):
            seen_names.add(abbr)
            found.append(
                FoundCompound(
                    name=abbr,
                    smiles=info.get("smiles", ""),
                    confidence=0.9,
                )
            )

    # Also match unicode variants (e.g., Pd₂(dba)₃ vs Pd2(dba)3)
    # Normalize subscript characters in text for matching
    _subscript_map = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    normalized = text.translate(_subscript_map)
    for abbr, info in reagents.items():
        escaped = re.escape(abbr)
        # G-R2-2 fix: case-insensitive matching for OCR-normalized text
        if abbr not in seen_names and re.search(
            r"(?<![a-zA-Z])" + escaped + r"(?![a-zA-Z])", normalized, re.IGNORECASE
        ):
            seen_names.add(abbr)
            found.append(
                FoundCompound(
                    name=abbr,
                    smiles=info.get("smiles", ""),
                    confidence=0.85,
                )
            )

    # Match common reagent formula patterns not in domain library
    # e.g., NaOH, KOH, HCl — try matching against normalized text
    for m in _RE_FORMULA.finditer(normalized):
        candidate = m.group(1)
        if candidate in reagents and candidate not in seen_names:
            info = reagents[candidate]
            seen_names.add(candidate)
            found.append(
                FoundCompound(
                    name=candidate,
                    smiles=info.get("smiles", ""),
                    confidence=0.85,
                )
            )

    # Match numbered compound references
    for m in _RE_COMPOUND_REF.finditer(text):
        ref_name = f"{m.group(1).lower()} {m.group(2)}"
        if ref_name not in seen_names:
            seen_names.add(ref_name)
            found.append(
                FoundCompound(
                    name=ref_name,
                    smiles="",
                    confidence=0.7,
                )
            )

    return found


# ---------------------------------------------------------------------------
# L3: Reaction role classifier (AC-20.3)
# ---------------------------------------------------------------------------

# Categories that map to "reagent" role in scheme.json
_REAGENT_CATEGORIES = frozenset(
    {
        "catalyst",
        "ligand",
        "base",
        "acid",
        "oxidant",
        "reductant",
        "reagent",
        "protecting_group",
    }
)
_SOLVENT_CATEGORIES = frozenset({"solvent"})
_COUPLING_CATEGORIES = frozenset({"coupling_partner"})


def _call_claude_role_api(compounds: list[FoundCompound], text: str) -> dict:
    """Call Claude API to classify compound roles in a reaction.

    Reuses the same API infrastructure as ai_intent (Phase 5).
    Returns dict with keys: reactants, products, reagents (each a list of dicts).
    """
    import json
    import os

    import requests

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    compound_list = ", ".join(c.name for c in compounds)
    system_prompt = (
        "You are a chemistry assistant. Given a list of chemical compounds found "
        "in experimental text, classify each compound's role in the reaction. "
        "Respond with ONLY valid JSON (no markdown fences). "
        "The JSON must have three keys: 'reactants', 'products', 'reagents'. "
        "Each is a list of objects with a 'name' field matching the input names exactly."
    )
    user_prompt = (
        f"Compounds: {compound_list}\n\n"
        f"Context: {text}\n\n"
        "Classify each compound as reactant, product, or reagent."
    )

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }

    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=payload,
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Claude API returned status {resp.status_code}")

    data = resp.json()
    content_list = data.get("content", [])
    content = content_list[0].get("text", "") if content_list else ""

    # Strip markdown fences if present
    from chemscheme.parsers.ai_intent import _strip_markdown_fences

    content = _strip_markdown_fences(content)

    result: dict = json.loads(content)
    return result


def classify_roles_ai(compounds: list[FoundCompound], text: str) -> ClassifiedReaction:
    """Use Claude API to classify reaction roles (fallback).

    This is the AI-assisted path when rule-based classification is ambiguous.
    Calls Claude API for role classification; falls back to all-reactants on failure.
    """
    logger.info("AI role classification requested for %d compounds", len(compounds))

    try:
        api_result = _call_claude_role_api(compounds, text)
    except Exception:
        logger.warning("AI role classification failed, falling back to all-reactants")
        return ClassifiedReaction(reactants=compounds, products=[], reagents=[])

    if not isinstance(api_result, dict):
        logger.warning("AI returned non-dict, falling back to all-reactants")
        return ClassifiedReaction(reactants=compounds, products=[], reagents=[])

    # Validate API response structure (G2/C2 fix)
    compound_by_name = {c.name: c for c in compounds}
    reactants: list[FoundCompound] = []
    products: list[FoundCompound] = []
    reagents: list[FoundCompound] = []
    classified_names: set[str] = set()

    for role, target_list in [
        ("reactants", reactants),
        ("products", products),
        ("reagents", reagents),
    ]:
        role_entries = api_result.get(role) or []  # Guard against None (G2)
        if not isinstance(role_entries, list):
            logger.warning("AI returned non-list for %s, falling back", role)
            return ClassifiedReaction(reactants=compounds, products=[], reagents=[])
        for entry in role_entries:
            name = entry.get("name", "") if isinstance(entry, dict) else ""
            if name in compound_by_name and name not in classified_names:  # dedup (C2)
                target_list.append(compound_by_name[name])
                classified_names.add(name)

    # Any compounds not classified by AI default to reactants
    for c in compounds:
        if c.name not in classified_names:
            reactants.append(c)

    return ClassifiedReaction(reactants=reactants, products=products, reagents=reagents)


def classify_roles(
    compounds: list[FoundCompound],
    text: str,
    use_ai: bool = True,
) -> ClassifiedReaction:
    """Assign roles to identified compounds.

    Rule-based first pass using domain library categories.
    Falls back to Claude API when ambiguous and use_ai=True.
    """
    if not compounds:
        return ClassifiedReaction()

    reagents_db = _get_reagent_abbrevs()
    conditions = extract_conditions(text)

    reactants: list[FoundCompound] = []
    products: list[FoundCompound] = []
    reagents: list[FoundCompound] = []
    unclassified: list[FoundCompound] = []

    for compound in compounds:
        # Normalize unicode subscripts for DB lookup
        _sub_map = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
        normalized_name = compound.name.translate(_sub_map)
        info = reagents_db.get(compound.name, {}) or reagents_db.get(normalized_name, {})
        category = info.get("category", "")

        if category in _SOLVENT_CATEGORIES:
            # Solvents go into conditions, not any role list (G3 fix: backfill)
            if conditions.solvent is None:
                conditions.solvent = compound.name
            continue
        elif category in _REAGENT_CATEGORIES:
            reagents.append(compound)
        elif category in _COUPLING_CATEGORIES:
            # Coupling partners are typically reactants
            reactants.append(compound)
        elif _is_product_marker(compound.name, text):
            products.append(compound)
        else:
            unclassified.append(compound)

    # If we have unclassified compounds and AI is enabled, try AI
    if unclassified and use_ai:
        ai_result = classify_roles_ai(unclassified, text)
        reactants.extend(ai_result.reactants)
        products.extend(ai_result.products)
        reagents.extend(ai_result.reagents)
    else:
        # Without AI, unclassified compounds default to reactants
        reactants.extend(unclassified)

    return ClassifiedReaction(
        reactants=reactants,
        products=products,
        reagents=reagents,
        conditions=conditions,
    )


def _is_product_marker(name: str, text: str) -> bool:
    """Check if a compound is marked as a product in the text.

    Uses direction-aware patterns: "to give X", "afforded X", "X (85% yield)".
    G4/C8 fix: avoid generic forward-window that misclassifies reactants.
    """
    name_lower = name.lower()
    text_lower = text.lower()

    # Pattern: "product X"
    if name_lower.startswith("product"):
        return True

    # Direction-aware: "to give/afford/obtained <name>" (R2-04: inflections + word boundary)
    give_re = re.compile(
        r"(?:to\s+)?(?:giv(?:e|es|ing)|afford(?:ed|s|ing)?|obtain(?:ed|s|ing)?"
        r"|yield(?:ed|s|ing)?|produc(?:e|ed|es|ing)|furnish(?:ed|es|ing)?)\s+"
        r"(?:the\s+)?(?:desired\s+)?(?:target\s+)?"
        + re.escape(name_lower)
        + r"(?!\w)",  # NEW-3: word boundary
    )
    if give_re.search(text_lower):
        return True

    # Direction-aware: "<name> ... (XX% yield)" within 40 chars after name (C1-R2 fix)
    for m in re.finditer(re.escape(name_lower) + r"(?!\w)", text_lower):
        after = text_lower[m.end() : m.end() + 40]
        if re.search(r"\(\s*\d+\s*%\s*yield\s*\)", after):
            return True
        if re.search(r"in\s+\d+\s*%\s*yield", after):
            return True

    return False


# ---------------------------------------------------------------------------
# L4: Text-to-scheme assembler (AC-20.4)
# ---------------------------------------------------------------------------

# Step delimiter patterns
_RE_STEP_SPLIT = re.compile(r"(?:Step\s+\d+\s*[:.]\s*)", re.IGNORECASE)


def _split_steps(text: str) -> list[str]:
    """Split multi-step procedures into individual paragraphs."""
    # Try step-numbered format first
    parts = _RE_STEP_SPLIT.split(text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) > 1:
        return parts
    # Single paragraph = single step
    return [text] if text.strip() else []


def extract_reactions_from_text(
    text: str,
    use_ai: bool = True,
) -> dict:
    """Extract structured reaction data from text into scheme.json format.

    Args:
        text: Experimental text (paragraph or multi-step procedure).
        use_ai: Whether to use Claude API for ambiguous role assignment.

    Returns:
        Dict compatible with scheme.json: {"steps": [...], "warnings": [...]}.
    """
    if not text or not text.strip():
        return {"schema_version": "1", "steps": [], "warnings": ["Empty input text"]}

    paragraphs = _split_steps(text)
    steps: list[dict] = []
    warnings: list[str] = []

    for i, para in enumerate(paragraphs):
        # L1: Extract conditions
        conditions = extract_conditions(para)

        # L2: Find compounds
        compounds = find_compounds(para)

        if not compounds:
            # No chemistry found in this paragraph
            continue

        # L3: Classify roles
        classified = classify_roles(compounds, para, use_ai=use_ai)

        # NEW-1 fix: use classified.conditions (may have backfilled solvent from L3)
        cond = classified.conditions or conditions

        # L4: Assemble step
        # R2-01 fix: only put SMILES in reactants/products, names go to compound_labels
        resolved_reactants = [c.smiles for c in classified.reactants if c.smiles]
        resolved_products = [c.smiles for c in classified.products if c.smiles]
        compound_labels: dict[str, str] = {}
        for c in [*classified.reactants, *classified.products]:
            if c.smiles:
                compound_labels[c.smiles] = c.name
            else:
                warnings.append(f"Unresolved compound: {c.name!r} (no SMILES)")

        step: dict = {
            "id": f"step_{i + 1}",
            "reactants": resolved_reactants,
            "products": resolved_products,
            "arrow_above": "",  # C3 fix: always present (schema required field)
        }

        if compound_labels:
            step["compound_labels"] = compound_labels

        # Emit structured fields for renderer compatibility (C5 fix)
        if classified.reagents:
            step["reagents"] = [c.name for c in classified.reagents]
        if cond.solvent:
            step["solvents"] = [cond.solvent]
        if cond.temperature:
            step["temperature"] = cond.temperature
        if cond.time:
            step["time"] = cond.time
        if cond.atmosphere:
            step["atmosphere"] = cond.atmosphere

        # Also build arrow_above as human-readable summary
        above_parts: list[str] = []
        if classified.reagents:
            above_parts.extend(c.name for c in classified.reagents)
        if cond.temperature:
            above_parts.append(cond.temperature)
        if cond.solvent:
            above_parts.append(cond.solvent)
        if cond.atmosphere:
            above_parts.append(cond.atmosphere)
        step["arrow_above"] = ", ".join(above_parts)

        if conditions.time:
            step["arrow_below"] = conditions.time
        if conditions.yield_pct:
            step["yield"] = conditions.yield_pct

        # C2-R2 fix: skip steps where ALL compounds are solvents/reagents only
        has_reactant_or_product = bool(classified.reactants or classified.products)
        if not has_reactant_or_product:
            warnings.append(f"Step {i + 1}: no reactants or products identified, skipped")
            continue

        # C3/R2 fix: skip steps where no SMILES resolved (would violate schema)
        if not resolved_reactants and not resolved_products:
            warnings.append(f"Step {i + 1}: no resolved SMILES, skipped")
            continue

        steps.append(step)

    if not steps and text.strip():
        warnings.append("No reaction data found in text")

    return {"schema_version": "1", "steps": steps, "warnings": warnings}
