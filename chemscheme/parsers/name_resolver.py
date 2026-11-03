"""Name resolution: compound name -> SMILES via local alias, OPSIN, PubChem.

Resolution order: local alias -> OPSIN -> PubChem.
All functions return SmilesResult (or None for individual resolvers on failure).
"""

import importlib.resources
import json
import logging
import threading
from pathlib import Path
from urllib.parse import quote as url_quote

import requests
from rdkit import Chem  # type: ignore[import-untyped]

from chemscheme.models import SmilesResult

_logger = logging.getLogger(__name__)

# Fallback for source-tree development (G6/C7)
_LEGACY_ALIASES_PATH = Path(__file__).parent.parent.parent / "data" / "reagent_aliases.json"
# G1 fix: use .smi extension to get plain SMILES (not HTML/JSON)
_OPSIN_URL = "https://opsin.ch.cam.ac.uk/opsin/{name}.smi"
_PUBCHEM_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES/JSON"
)
# MR3-5: descriptive User-Agent to avoid throttling by academic APIs
_REQUEST_HEADERS = {"User-Agent": "ChemScheme/0.1.0 (chemistry-cli)"}

# Cache loaded aliases: abbreviation -> {"name": ..., "smiles": ...}
_alias_cache: dict[str, dict[str, str]] | None = None
_alias_lock = threading.Lock()


def _load_aliases_text() -> str:
    """Load reagent_aliases.json text via importlib.resources with fallback."""
    try:
        ref = importlib.resources.files("chemscheme.data").joinpath("reagent_aliases.json")
        return ref.read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, TypeError):
        _logger.debug("importlib.resources failed, falling back to legacy path")

    if _LEGACY_ALIASES_PATH.exists():
        return _LEGACY_ALIASES_PATH.read_text(encoding="utf-8")

    raise FileNotFoundError("reagent_aliases.json not found. Expected in chemscheme/data/ or data/")


def _load_aliases() -> dict[str, dict[str, str]]:
    """Load reagent_aliases.json abbreviations.

    Supports both legacy format (abbr -> name string)
    and new format (abbr -> {"name": ..., "smiles": ...}).
    """
    global _alias_cache  # noqa: PLW0603
    if _alias_cache is not None:
        return _alias_cache
    with _alias_lock:  # MR4-14: thread-safe initialization
        if _alias_cache is not None:
            return _alias_cache  # double-check after acquiring lock
        try:
            data = json.loads(_load_aliases_text())
        except (FileNotFoundError, json.JSONDecodeError) as e:
            # Graceful fallback: empty aliases (MR5R2-G4)
            _logger.warning("Failed to load reagent aliases: %s", e)
            _alias_cache = {}
            return _alias_cache
        raw = data.get("abbreviations", {})
        normalized: dict[str, dict[str, str]] = {}
        for key, val in raw.items():
            if key.startswith("_"):  # Skip metadata keys like _comment_* (C4-R2)
                continue
            if isinstance(val, str):
                normalized[key] = {"name": val}
            else:
                normalized[key] = val
        _alias_cache = normalized
    return _alias_cache


def _canonicalize_smiles(smiles: str) -> str | None:
    """Canonicalize SMILES via RDKit. Returns None if RDKit can't parse (MR3-2)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    result: str = Chem.MolToSmiles(mol)
    return result


def resolve_local_alias(name: str) -> SmilesResult | None:
    """Look up abbreviation in reagent_aliases.json.

    If the alias entry has a pre-cached 'smiles' field, returns immediately
    without any network call. Otherwise falls back to OPSIN/PubChem.

    Returns SmilesResult if found, None otherwise.
    Case-sensitive lookup.
    """
    aliases = _load_aliases()
    entry = aliases.get(name)
    if entry is None:
        return None

    # If pre-cached SMILES exists, try it first (no network needed)
    cached_smiles = entry.get("smiles")
    if cached_smiles:
        canonical = _canonicalize_smiles(cached_smiles)
        if canonical is not None:
            return SmilesResult(
                smiles=canonical,
                confidence=0.95,
                source="local_alias",
                warnings=[],
            )
        # Bad cached SMILES — fall through to IUPAC name resolution (MR4.1-P2c)
        _logger.warning(
            "Local alias '%s' has invalid SMILES: %s, trying IUPAC name", name, cached_smiles
        )

    # Fall back to resolving the IUPAC name via network
    iupac_name = entry.get("name", "")
    if not iupac_name:
        return None

    result = resolve_via_opsin(iupac_name)
    if result is not None:
        return SmilesResult(
            smiles=result.smiles,
            confidence=0.95,
            source="local_alias",
            warnings=[],
        )

    result = resolve_via_pubchem(iupac_name)
    if result is not None:
        return SmilesResult(
            smiles=result.smiles,
            confidence=0.9,
            source="local_alias",
            warnings=[],
        )

    return None


def resolve_via_opsin(name: str) -> SmilesResult | None:
    """Resolve compound name via OPSIN web service.

    Returns SmilesResult on success, None on failure.
    """
    try:
        url = _OPSIN_URL.format(name=url_quote(name, safe=""))
        resp = requests.get(url, headers=_REQUEST_HEADERS, timeout=10)
        if resp.status_code != 200:
            return None

        smiles = resp.text.strip()
        if not smiles:
            return None

        canonical = _canonicalize_smiles(smiles)
        if canonical is None:
            _logger.warning("OPSIN returned unparseable SMILES for '%s': %s", name, smiles)
            return None
        return SmilesResult(
            smiles=canonical,
            confidence=0.9,
            source="opsin",
            warnings=[],
        )
    except Exception:
        return None


def resolve_via_pubchem(name: str) -> SmilesResult | None:
    """Resolve compound name via PubChem REST API.

    Returns SmilesResult on success, None on failure.
    """
    try:
        url = _PUBCHEM_URL.format(name=url_quote(name, safe=""))
        resp = requests.get(url, headers=_REQUEST_HEADERS, timeout=10)
        if resp.status_code != 200:
            return None

        data = resp.json()
        props = data.get("PropertyTable", {}).get("Properties", [])
        if not props:
            return None

        smiles = props[0].get("CanonicalSMILES", "")
        if not smiles:
            return None

        canonical = _canonicalize_smiles(smiles)
        if canonical is None:
            _logger.warning("PubChem returned unparseable SMILES for '%s': %s", name, smiles)
            return None
        return SmilesResult(
            smiles=canonical,
            confidence=0.85,
            source="pubchem",
            warnings=[],
        )
    except Exception:
        return None


def resolve_name(name: str) -> SmilesResult:
    """Resolve a compound name to SMILES via the full chain.

    Order: local alias -> OPSIN -> PubChem.
    Always returns SmilesResult (confidence=0 on total failure).
    """
    # 1. Local alias
    result = resolve_local_alias(name)
    if result is not None:
        return result

    # 2. OPSIN
    result = resolve_via_opsin(name)
    if result is not None:
        return result

    # 3. PubChem
    result = resolve_via_pubchem(name)
    if result is not None:
        return result

    # All failed
    return SmilesResult(
        smiles="",
        confidence=0.0,
        source="none",
        warnings=[f"Could not resolve '{name}' via local alias, OPSIN, or PubChem"],
    )
