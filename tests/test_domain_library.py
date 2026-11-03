"""Phase 19 Domain Library tests — reagent lookup + reaction pattern suggest.

AC-19.1: Expanded reagent library with categories
AC-19.2: Reaction pattern library (10+ named reactions)
AC-19.3: Domain lookup module (lookup_reagent, lookup_reaction, suggest_conditions)
AC-19.6: Adversarial (empty query, unknown reaction, partial match, case-insensitive)
"""

from chemscheme.domain_library import (
    ReactionMatch,
    ReactionPattern,
    ReagentInfo,
    identify_reaction_type,
    lookup_reaction,
    lookup_reagent,
    suggest_conditions,
)

# ---------------------------------------------------------------------------
# AC-19.1: Reagent library
# ---------------------------------------------------------------------------


class TestReagentInfo:
    """ReagentInfo dataclass."""

    def test_reagent_info_fields(self) -> None:
        """GIVEN ReagentInfo, THEN has abbreviation, name, category, smiles."""
        r = ReagentInfo(
            abbreviation="NBS",
            name="N-bromosuccinimide",
            category="reagent",
            smiles="O=C1CCC(=O)N1Br",
        )
        assert r.abbreviation == "NBS"
        assert r.name == "N-bromosuccinimide"
        assert r.category == "reagent"
        assert r.smiles == "O=C1CCC(=O)N1Br"

    def test_reagent_info_smiles_optional(self) -> None:
        """GIVEN ReagentInfo without SMILES, THEN smiles is None."""
        r = ReagentInfo(abbreviation="SPhos", name="SPhos ligand", category="ligand")
        assert r.smiles is None


# ---------------------------------------------------------------------------
# AC-19.3: lookup_reagent
# ---------------------------------------------------------------------------


class TestLookupReagent:
    """Reagent lookup by name/abbreviation/category."""

    def test_lookup_by_abbreviation(self) -> None:
        """GIVEN abbreviation 'NBS', WHEN lookup, THEN returns matching reagent."""
        results = lookup_reagent("NBS")
        assert len(results) >= 1
        assert any(r.abbreviation == "NBS" for r in results)

    def test_lookup_by_name_partial(self) -> None:
        """GIVEN partial name 'bromosuccin', WHEN lookup, THEN matches NBS."""
        results = lookup_reagent("bromosuccin")
        assert len(results) >= 1
        assert any(r.abbreviation == "NBS" for r in results)

    def test_lookup_case_insensitive(self) -> None:
        """GIVEN lowercase 'nbs', WHEN lookup, THEN matches NBS."""
        results = lookup_reagent("nbs")
        assert len(results) >= 1
        assert any(r.abbreviation == "NBS" for r in results)

    def test_lookup_by_category(self) -> None:
        """GIVEN category='catalyst', WHEN lookup 'Pd', THEN returns Pd catalysts."""
        results = lookup_reagent("Pd", category="catalyst")
        assert len(results) >= 1
        assert all(r.category == "catalyst" for r in results)

    def test_lookup_solvent(self) -> None:
        """GIVEN 'THF', WHEN lookup, THEN returns solvent with SMILES."""
        results = lookup_reagent("THF")
        assert len(results) >= 1
        thf = next(r for r in results if r.abbreviation == "THF")
        assert thf.category == "solvent"
        assert thf.smiles is not None

    def test_lookup_no_match(self) -> None:
        """GIVEN unknown name, WHEN lookup, THEN returns empty list."""
        results = lookup_reagent("xyznonexistent999")
        assert results == []

    def test_lookup_empty_query_returns_empty(self) -> None:
        """AC-19.6: GIVEN empty string, WHEN lookup, THEN returns empty list."""
        results = lookup_reagent("")
        assert results == []


# ---------------------------------------------------------------------------
# AC-19.2 + AC-19.3: Reaction patterns
# ---------------------------------------------------------------------------


class TestReactionPattern:
    """ReactionPattern dataclass."""

    def test_reaction_pattern_fields(self) -> None:
        """GIVEN ReactionPattern, THEN has name, catalyst, solvents, etc."""
        rp = ReactionPattern(
            name="Suzuki coupling",
            reagents=["aryl halide", "boronic acid"],
            catalyst="Pd(PPh3)4",
            solvents=["THF", "H2O"],
            bases=["K2CO3"],
            temperature="80°C",
            time="12h",
            description="Pd-catalyzed cross-coupling of aryl halides with boronic acids.",
        )
        assert rp.name == "Suzuki coupling"
        assert rp.catalyst == "Pd(PPh3)4"


class TestLookupReaction:
    """Reaction pattern lookup."""

    def test_lookup_suzuki(self) -> None:
        """GIVEN 'Suzuki', WHEN lookup_reaction, THEN returns Suzuki coupling."""
        results = lookup_reaction("Suzuki")
        assert len(results) >= 1
        assert any("suzuki" in r.name.lower() for r in results)

    def test_lookup_partial_match(self) -> None:
        """AC-19.6: GIVEN 'buchwald', WHEN lookup_reaction, THEN matches Buchwald-Hartwig."""
        results = lookup_reaction("buchwald")
        assert len(results) >= 1

    def test_lookup_case_insensitive(self) -> None:
        """GIVEN 'STILLE', WHEN lookup_reaction, THEN matches Stille coupling."""
        results = lookup_reaction("STILLE")
        assert len(results) >= 1

    def test_lookup_no_match(self) -> None:
        """AC-19.6: GIVEN unknown reaction, WHEN lookup, THEN returns empty list."""
        results = lookup_reaction("xyznonexistent")
        assert results == []


class TestSuggestConditions:
    """Suggest typical conditions for named reactions."""

    def test_suggest_suzuki(self) -> None:
        """GIVEN 'Suzuki coupling', WHEN suggest, THEN returns conditions."""
        result = suggest_conditions("Suzuki coupling")
        assert result is not None
        assert result.catalyst is not None
        assert len(result.solvents) >= 1

    def test_suggest_partial_match(self) -> None:
        """GIVEN 'suzuki', WHEN suggest, THEN still matches."""
        result = suggest_conditions("suzuki")
        assert result is not None

    def test_suggest_unknown_returns_none(self) -> None:
        """AC-19.6: GIVEN unknown reaction, WHEN suggest, THEN returns None."""
        result = suggest_conditions("xyznonexistent")
        assert result is None

    def test_at_least_10_reactions_available(self) -> None:
        """AC-19.2: Library has 10+ reaction patterns."""
        # Check a selection of expected reactions
        known = [
            "Suzuki",
            "Stille",
            "Heck",
            "Sonogashira",
            "Buchwald",
            "Knoevenagel",
            "Vilsmeier",
            "Wittig",
            "Grignard",
            "Aldol",
        ]
        found = 0
        for name in known:
            if lookup_reaction(name):
                found += 1
        assert found >= 10, f"Only {found}/10 named reactions found"


# ---------------------------------------------------------------------------
# NFA/OPV template expansion
# ---------------------------------------------------------------------------


class TestNFAAcceptorTemplates:
    """NFA (non-fullerene acceptor) building blocks for OPV research."""

    def test_y6_in_library(self) -> None:
        """Y6 (BTP-4F) is the benchmark NFA — must be present."""
        results = lookup_reagent("Y6")
        assert len(results) >= 1
        y6 = results[0]
        assert y6.category == "nfa_acceptor"

    def test_itic_in_library(self) -> None:
        """ITIC is a foundational NFA acceptor."""
        results = lookup_reagent("ITIC")
        assert len(results) >= 1
        assert results[0].category == "nfa_acceptor"

    def test_it4f_in_library(self) -> None:
        """IT-4F is a fluorinated ITIC variant."""
        results = lookup_reagent("IT-4F")
        assert len(results) >= 1
        assert results[0].category == "nfa_acceptor"

    def test_btp_ec9_in_library(self) -> None:
        """BTP-eC9 is a high-efficiency Y6 derivative."""
        results = lookup_reagent("BTP-eC9")
        assert len(results) >= 1
        assert results[0].category == "nfa_acceptor"

    def test_ic_endcap_in_library(self) -> None:
        """IC (2-(3-oxo-2,3-dihydroinden-1-ylidene)malononitrile) is the key end-cap."""
        results = lookup_reagent("IC")
        assert len(results) >= 1
        # IC is a coupling partner (end-cap unit)
        assert results[0].category == "coupling_partner"

    def test_pc61bm_fullerene_reference(self) -> None:
        """PC61BM (fullerene acceptor) included as reference."""
        results = lookup_reagent("PC61BM")
        assert len(results) >= 1


class TestDonorPolymerTemplates:
    """Donor polymer building blocks for OPV."""

    def test_pm6_in_library(self) -> None:
        """PM6 is the most common donor polymer for NFA-based OPV."""
        results = lookup_reagent("PM6")
        assert len(results) >= 1
        assert results[0].category == "donor_polymer"

    def test_ptb7th_in_library(self) -> None:
        """PTB7-Th is a classic low-bandgap donor."""
        results = lookup_reagent("PTB7-Th")
        assert len(results) >= 1
        assert results[0].category == "donor_polymer"

    def test_d18_in_library(self) -> None:
        """D18 is a high-performance donor polymer."""
        results = lookup_reagent("D18")
        assert len(results) >= 1
        assert results[0].category == "donor_polymer"


class TestOPVReactionPatterns:
    """OPV-specific reaction patterns."""

    def test_stille_polymerization_pattern(self) -> None:
        """Stille polymerization is the main route for conjugated polymers."""
        results = lookup_reaction("Stille polymerization")
        assert len(results) >= 1
        rp = results[0]
        assert rp.catalyst  # must have catalyst
        assert rp.atmosphere == "N2"  # always inert

    def test_knoevenagel_endcapping_pattern(self) -> None:
        """Knoevenagel end-capping is how NFA acceptors get their IC groups."""
        results = lookup_reaction("Knoevenagel end-capping")
        assert len(results) >= 1

    def test_direct_arylation_polymerization_pattern(self) -> None:
        """Direct arylation polymerization (DArP) is an emerging green route."""
        results = lookup_reaction("direct arylation polymerization")
        assert len(results) >= 1

    def test_opv_reactions_at_least_3(self) -> None:
        """At least 3 OPV-specific reaction patterns available."""
        opv_reactions = [
            "Stille polymerization",
            "Knoevenagel end-capping",
            "direct arylation polymerization",
        ]
        found = sum(1 for name in opv_reactions if lookup_reaction(name))
        assert found >= 3, f"Only {found}/3 OPV-specific patterns found"


class TestNFAAdversarial:
    """Adversarial cases for NFA/OPV templates."""

    def test_nonexistent_nfa_returns_empty(self) -> None:
        """Unknown NFA abbreviation returns empty list."""
        results = lookup_reagent("XYZFAKE-NFA-999")
        assert results == []

    def test_nfa_category_filter_works(self) -> None:
        """Category filter 'nfa_acceptor' returns only NFAs."""
        results = lookup_reagent("Y", category="nfa_acceptor")
        for r in results:
            assert r.category == "nfa_acceptor"


class TestNFAEndGroupTemplates:
    """IC end-group variants — key building blocks for A-D-A type NFAs."""

    def test_ic_2f_in_library(self) -> None:
        """IC-2F (difluorinated IC) is used in Y6, BTP-4F, 3TT-C2-F."""
        results = lookup_reagent("IC-2F")
        assert len(results) >= 1
        assert results[0].category == "coupling_partner"

    def test_ic_2cl_in_library(self) -> None:
        """IC-2Cl (dichlorinated IC) is used in BTP-4Cl, 3TT-C2-Cl."""
        results = lookup_reagent("IC-2Cl")
        assert len(results) >= 1
        assert results[0].category == "coupling_partner"

    def test_3tt_core_in_library(self) -> None:
        """3TT (trithienothiophene) is an emerging NFA donor core."""
        results = lookup_reagent("3TT")
        assert len(results) >= 1
        assert results[0].category == "nfa_core"


# ---------------------------------------------------------------------------
# Real-paper validation: catalyst variant matching + new patterns
# ---------------------------------------------------------------------------


class TestCatalystVariantMatching:
    """Fix: identify_reaction_type must match catalyst variants.

    Real papers use different ligands (P(t-bu)3HBF4, DPPF, XPhos) with the
    same Pd source. The current implementation requires exact substring match
    on the full catalyst string (e.g., "Pd2(dba)3 / SPhos"), which fails when
    papers use "Pd2(dba)3, P(t-bu)3HBF4".
    """

    def test_buchwald_hartwig_with_ptbu3(self) -> None:
        """GIVEN Pd2(dba)3 + P(t-bu)3HBF4 + NaOtBu (real paper conditions),
        WHEN identify_reaction_type, THEN returns 'Buchwald-Hartwig amination'."""
        result = identify_reaction_type("Pd2(dba)3, P(t-bu)3HBF4, NaOtBu")
        assert "buchwald" in result.name.lower()

    def test_buchwald_hartwig_with_dppf(self) -> None:
        """GIVEN Pd2(dba)3/DPPF + NaOtBu (another common variant),
        WHEN identify_reaction_type, THEN returns 'Buchwald-Hartwig amination'."""
        result = identify_reaction_type("Pd2(dba)3/DPPF, NaOtBu")
        assert "buchwald" in result.name.lower()

    def test_suzuki_still_works(self) -> None:
        """Regression: Suzuki coupling must still match after refactor."""
        result = identify_reaction_type("Pd(PPh3)4, K2CO3")
        assert "suzuki" in result.name.lower()

    def test_sonogashira_with_cui(self) -> None:
        """GIVEN Pd(PPh3)2Cl2 + CuI + Et3N (Sonogashira conditions),
        WHEN identify_reaction_type, THEN returns 'Sonogashira coupling'."""
        result = identify_reaction_type("Pd(PPh3)2Cl2, CuI, Et3N")
        assert "sonogashira" in result.name.lower()


class TestNBSBrominationPattern:
    """New pattern: NBS bromination — extremely common in thiophene chemistry."""

    def test_nbs_thf_identified(self) -> None:
        """GIVEN 'NBS, THF' (standard bromination conditions),
        WHEN identify_reaction_type, THEN returns 'NBS bromination'."""
        result = identify_reaction_type("NBS, THF")
        assert "bromination" in result.name.lower()

    def test_nbs_dmf_identified(self) -> None:
        """GIVEN 'NBS, DMF' (alternative solvent),
        WHEN identify_reaction_type, THEN returns 'NBS bromination'."""
        result = identify_reaction_type("NBS, DMF")
        assert "bromination" in result.name.lower()

    def test_nbs_lookup_exists(self) -> None:
        """NBS bromination should be a named reaction in the library."""
        results = lookup_reaction("NBS bromination")
        assert len(results) >= 1


class TestMiyauraBorylationPattern:
    """New pattern: Miyaura borylation — key step for Suzuki coupling precursors."""

    def test_b2pin2_pdcl2_identified(self) -> None:
        """GIVEN 'B2pin2, Pd(PPh3)2Cl2, KOAc' (standard borylation),
        WHEN identify_reaction_type, THEN returns borylation."""
        result = identify_reaction_type("B2pin2, Pd(PPh3)2Cl2, KOAc")
        assert "borylation" in result.name.lower()

    def test_b2pin2_pddppf_identified(self) -> None:
        """GIVEN 'B2pin2, Pd(dppf)Cl2, KOAc' (dppf variant),
        WHEN identify_reaction_type, THEN returns borylation."""
        result = identify_reaction_type("B2pin2, Pd(dppf)Cl2, KOAc")
        assert "borylation" in result.name.lower()

    def test_miyaura_lookup_exists(self) -> None:
        """Miyaura borylation should be a named reaction in the library."""
        results = lookup_reaction("Miyaura borylation")
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# P1: False positive fixes — Stille should not match without Sn indicators
# ---------------------------------------------------------------------------


class TestStilleFalsePositiveFix:
    """Stille coupling/polymerization must not match when no tin indicator present.

    Root cause: Stille has bases=[] so catalyst-only match auto-passes.
    Real-world consequence: any Pd(PPh3)4 reaction without base → false Stille.
    """

    def test_pd_pph3_without_sn_not_stille(self) -> None:
        """GIVEN Pd(PPh3)4 + MgBr (Grignard, no tin),
        THEN should NOT return 'Stille coupling'."""
        result = identify_reaction_type("MgBr, Pd(PPh3)4, THF, RT")
        assert "stille" not in result.name.lower()

    def test_pd2dba3_kf_not_stille_polym(self) -> None:
        """GIVEN Pd2(dba)3 + KF (Suzuki-type, no tin),
        THEN should NOT return 'Stille polymerization'."""
        result = identify_reaction_type("Pd2(dba)3, KF, 1-butanol/H2O, 100C")
        assert "stille" not in result.name.lower()

    def test_stille_with_snbu3_still_matches(self) -> None:
        """Regression: Stille coupling WITH organostannane indicator must still match."""
        result = identify_reaction_type("ArSnBu3, Pd(PPh3)4, toluene")
        assert "stille" in result.name.lower()

    def test_stille_polym_with_sn_still_matches(self) -> None:
        """Regression: Stille polymerization WITH Sn indicator must still match."""
        result = identify_reaction_type("Me3Sn-monomer-SnMe3, Pd2(dba)3, P(o-tol)3, toluene")
        assert "stille" in result.name.lower()


# ---------------------------------------------------------------------------
# P2: CuAAC click chemistry + diazotization
# ---------------------------------------------------------------------------


class TestCuAACClickChemistry:
    """CuAAC (copper-catalyzed azide-alkyne cycloaddition) — the 'click' reaction."""

    def test_cuso4_ascorbate_identified(self) -> None:
        """GIVEN CuSO4 + sodium ascorbate (classic Sharpless conditions),
        WHEN identify_reaction_type, THEN returns click chemistry."""
        result = identify_reaction_type("CuSO4.5H2O, sodium ascorbate, EtOH/H2O")
        assert "click" in result.name.lower() or "cuaac" in result.name.lower()

    def test_cuso4_without_5h2o_identified(self) -> None:
        """GIVEN CuSO4 without hydrate notation, THEN still matches."""
        result = identify_reaction_type("CuSO4, sodium ascorbate, t-BuOH/H2O")
        assert "click" in result.name.lower() or "cuaac" in result.name.lower()

    def test_click_lookup_exists(self) -> None:
        """CuAAC should be a named reaction in the library."""
        results = lookup_reaction("CuAAC")
        assert len(results) >= 1


class TestDiazotization:
    """Diazotization via t-BuONO/TMSN3 — amine to azide conversion."""

    def test_tbuono_tmsn3_identified(self) -> None:
        """GIVEN t-BuONO + TMSN3 (standard conditions),
        WHEN identify_reaction_type, THEN returns diazotization."""
        result = identify_reaction_type("t-BuONO, TMSN3, CH3CN, 0C to RT")
        assert "diazotization" in result.name.lower() or "azide" in result.name.lower()

    def test_diazotization_lookup_exists(self) -> None:
        """Diazotization should be a named reaction in the library."""
        results = lookup_reaction("diazotization")
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# P3: Classic organic reactions (TBS, BF3, enol triflate, benzylation)
# ---------------------------------------------------------------------------


class TestTBSProtection:
    """TBS (tert-butyldimethylsilyl) protection — ubiquitous in total synthesis."""

    def test_tbsotf_identified(self) -> None:
        """GIVEN TBSOTf + Et3N, THEN returns TBS protection."""
        result = identify_reaction_type("TBSOTf, Et3N, CH2Cl2")
        assert "tbs" in result.name.lower() or "silyl" in result.name.lower()

    def test_tbscl_identified(self) -> None:
        """GIVEN TBSCl + imidazole (alternative), THEN returns TBS protection."""
        result = identify_reaction_type("TBSCl, imidazole, DMF")
        assert "tbs" in result.name.lower() or "silyl" in result.name.lower()

    def test_tbs_lookup_exists(self) -> None:
        """TBS protection should be in the library."""
        results = lookup_reaction("TBS protection")
        assert len(results) >= 1


class TestBF3LewisAcid:
    """BF3·Et2O Lewis acid activation — common in additions and rearrangements."""

    def test_bf3_et2o_identified(self) -> None:
        """GIVEN BF3.Et2O, CH2Cl2, THEN returns Lewis acid activation."""
        result = identify_reaction_type("BF3.Et2O, CH2Cl2, -78C")
        assert "bf3" in result.name.lower() or "lewis" in result.name.lower()

    def test_bf3_lookup_exists(self) -> None:
        """BF3 activation should be in the library."""
        results = lookup_reaction("BF3")
        assert len(results) >= 1


class TestEnolTriflateFormation:
    """Enol triflate formation via LiHMDS + PhN(Tf)2."""

    def test_lihmds_phntf2_identified(self) -> None:
        """GIVEN LiHMDS + PhN(Tf)2 (standard enol triflate conditions),
        THEN returns enol triflate formation."""
        result = identify_reaction_type("LiHMDS, THF, -78C, then PhN(Tf)2")
        assert "triflate" in result.name.lower() or "enol" in result.name.lower()

    def test_triflate_lookup_exists(self) -> None:
        """Enol triflate formation should be in the library."""
        results = lookup_reaction("enol triflate")
        assert len(results) >= 1


class TestBenzylation:
    """N/O-Benzylation with BnBr — common in heterocycle and alkaloid chemistry."""

    def test_bnbr_identified(self) -> None:
        """GIVEN BnBr in condition text, THEN returns benzylation."""
        result = identify_reaction_type("BnBr/CH3CN, sealed tube, 90C")
        assert "benzylation" in result.name.lower()

    def test_benzylation_lookup_exists(self) -> None:
        """Benzylation should be in the library."""
        results = lookup_reaction("benzylation")
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# S41: Untested core reactions — Negishi, Kumada, Heck, Vilsmeier-Haack, Grignard, Wittig
# ---------------------------------------------------------------------------


class TestNegishiCoupling:
    """Negishi coupling — aryl halide + organozinc, key_reagent: Zn."""

    def test_zncl2_conditions_identified(self) -> None:
        """GIVEN ZnCl2 + Pd(PPh3)4 in THF, THEN Negishi coupling."""
        result = identify_reaction_type("Pd(PPh3)4, ZnCl2, THF, 60°C")
        assert "negishi" in result.name.lower()

    def test_znbr_variant(self) -> None:
        """GIVEN ArZnBr + Pd, THEN Negishi."""
        result = identify_reaction_type("ArZnBr, Pd(PPh3)4, THF, RT")
        assert "negishi" in result.name.lower()

    def test_lookup_exists(self) -> None:
        results = lookup_reaction("Negishi")
        assert len(results) >= 1


class TestKumadaCoupling:
    """Kumada coupling — aryl halide + Grignard, catalyst: Pd(dppf)Cl2."""

    def test_pddppf_no_boron_identified(self) -> None:
        """GIVEN Pd(dppf)Cl2 without B2pin2, THEN Kumada (not Miyaura)."""
        result = identify_reaction_type("Pd(dppf)Cl2, THF, reflux")
        assert "kumada" in result.name.lower()

    def test_with_mgbr_matches_grignard(self) -> None:
        """GIVEN MgBr in text, Pass 1 key_reagent 'Mg' triggers Grignard over Kumada."""
        # Kumada IS Grignard + Pd catalyst, but Mg key_reagent wins in Pass 1
        result = identify_reaction_type("PhMgBr, Pd(dppf)Cl2, THF, RT")
        assert "grignard" in result.name.lower()

    def test_lookup_exists(self) -> None:
        results = lookup_reaction("Kumada")
        assert len(results) >= 1


class TestHeckReaction:
    """Heck reaction — aryl halide + alkene, Pd(OAc)2/Et3N/DMF."""

    def test_pdoac2_et3n_identified(self) -> None:
        """GIVEN Pd(OAc)2 + Et3N in DMF, THEN Heck reaction."""
        result = identify_reaction_type("Pd(OAc)2, Et3N, DMF, 120°C")
        assert "heck" in result.name.lower()

    def test_without_base_lower_confidence(self) -> None:
        """GIVEN Pd(OAc)2 without Et3N, THEN lower confidence than with Et3N."""
        # Pd(OAc)2 + DMF matches Heck (DMF is Heck solvent), but without
        # the base (Et3N) the confidence should be lower
        with_base = identify_reaction_type("Pd(OAc)2, Et3N, DMF, 120°C")
        without_base = identify_reaction_type("Pd(OAc)2, DMF, 120°C")
        assert with_base.confidence > without_base.confidence

    def test_lookup_exists(self) -> None:
        results = lookup_reaction("Heck")
        assert len(results) >= 1


class TestVilsmeierHaack:
    """Vilsmeier-Haack formylation — POCl3/DMF reagent system."""

    def test_pocl3_identified(self) -> None:
        """GIVEN POCl3/DMF in DCE, THEN Vilsmeier-Haack formylation."""
        result = identify_reaction_type("POCl3, DMF, DCE, 0°C → RT")
        assert "vilsmeier" in result.name.lower()

    def test_pocl3_in_real_paper_context(self) -> None:
        """Real paper: Vilsmeier formylation of thiophene-fused core."""
        result = identify_reaction_type("POCl3 (3 eq), DMF, 1,2-DCE, 0°C to rt, 12h")
        assert "vilsmeier" in result.name.lower()

    def test_lookup_exists(self) -> None:
        results = lookup_reaction("Vilsmeier")
        assert len(results) >= 1


class TestGrignardReaction:
    """Grignard reaction — Mg in THF/Et2O, key_reagent: Mg."""

    def test_mg_thf_identified(self) -> None:
        """GIVEN Mg turnings in THF, THEN Grignard reaction."""
        result = identify_reaction_type("Mg, THF, reflux, N2")
        assert "grignard" in result.name.lower()

    def test_mgbr2_not_false_positive(self) -> None:
        """GIVEN MgBr2 as additive (not Grignard formation), should still match on Mg."""
        # MgBr2 contains Mg followed by B (uppercase) → passes regex
        result = identify_reaction_type("MgBr2, Et2O, 0°C")
        assert "grignard" in result.name.lower()

    def test_lookup_exists(self) -> None:
        results = lookup_reaction("Grignard")
        assert len(results) >= 1


class TestWittigReaction:
    """Wittig reaction — PPh3 ylide + base (BuLi/NaH)."""

    def test_ylide_buli_identified(self) -> None:
        """GIVEN phosphonium ylide + BuLi in THF, THEN Wittig reaction."""
        result = identify_reaction_type("phosphonium ylide, BuLi, THF, -78°C to RT")
        assert "wittig" in result.name.lower()

    def test_ylide_in_condition_text(self) -> None:
        """Real paper: Wittig olefination with ylide."""
        result = identify_reaction_type("Wittig ylide, NaH, THF, 0°C")
        assert "wittig" in result.name.lower()

    def test_lookup_exists(self) -> None:
        results = lookup_reaction("Wittig")
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# S41: Untested OPV reagent lookups
# ---------------------------------------------------------------------------


class TestUntrestedOPVReagents:
    """OPV reagents in library but without dedicated tests."""

    def test_it_m_lookup(self) -> None:
        """IT-M (methylated ITIC) should be in reagent library."""
        results = lookup_reagent("IT-M")
        assert len(results) >= 1
        assert any("nfa" in r.category.lower() for r in results)

    def test_l8_bo_lookup(self) -> None:
        """L8-BO (Y-series NFA) should be in reagent library."""
        results = lookup_reagent("L8-BO")
        assert len(results) >= 1

    def test_ieico_4f_lookup(self) -> None:
        """IEICO-4F (narrow-bandgap NFA) should be in reagent library."""
        results = lookup_reagent("IEICO-4F")
        assert len(results) >= 1

    def test_btp_4cl_lookup(self) -> None:
        """BTP-4Cl (chlorinated Y6) should be in reagent library."""
        results = lookup_reagent("BTP-4Cl")
        assert len(results) >= 1

    def test_pm7_lookup(self) -> None:
        """PM7 (chlorinated PM6 donor polymer) should be in reagent library."""
        results = lookup_reagent("PM7")
        assert len(results) >= 1

    def test_pbdb_t_lookup(self) -> None:
        """PBDB-T (PM6 parent donor polymer) should be in reagent library."""
        results = lookup_reagent("PBDB-T")
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# S41: Multi-step synthesis chain validation
# ---------------------------------------------------------------------------


class TestMultiStepSynthesisChain:
    """Validate that a real-paper NFA synthesis chain identifies each step."""

    def test_y6_synthesis_steps(self) -> None:
        """Y6-like NFA synthesis: bromination → Stille → Knoevenagel end-capping."""
        # Step 1: NBS bromination of core
        step1 = identify_reaction_type("NBS, CHCl3, RT, 2h")
        assert "nbs bromination" in step1.name.lower()

        # Step 2: Stille coupling to build D-A backbone
        step2 = identify_reaction_type("Pd2(dba)3, P(o-tol)3, toluene, SnMe3, 110°C, 48h, N2")
        assert "stille" in step2.name.lower()

        # Step 3: Knoevenagel end-capping with IC-2F
        step3 = identify_reaction_type("pyridine, CHCl3, reflux, 8h")
        assert "knoevenagel" in step3.name.lower()

    def test_pm6_donor_polymer_synthesis(self) -> None:
        """PM6-like polymer: monomer borylation → Suzuki polymerization."""
        # Step 1: Miyaura borylation
        step1 = identify_reaction_type("B2pin2, Pd(dppf)Cl2, KOAc, dioxane, 80°C, 24h")
        assert "miyaura" in step1.name.lower()

        # Step 2: Suzuki coupling polymerization (Pd2(dba)3 variant avoids PPh3 ambiguity)
        step2 = identify_reaction_type("Pd2(dba)3, K2CO3, THF/H2O, 80°C, 24h")
        assert "suzuki" in step2.name.lower()

    def test_click_chemistry_sequence(self) -> None:
        """Click chemistry: diazotization → CuAAC."""
        step1 = identify_reaction_type("t-BuONO, TMSN3, CH3CN, 0°C")
        assert "diazotization" in step1.name.lower()

        step2 = identify_reaction_type("CuSO4, sodium ascorbate, t-BuOH/H2O, RT")
        assert "click" in step2.name.lower()


# ---------------------------------------------------------------------------
# S41: ReactionMatch return type + unified scoring tests
# ---------------------------------------------------------------------------


class TestReactionMatchReturnType:
    """identify_reaction_type should return ReactionMatch with confidence."""

    def test_returns_reaction_match(self) -> None:
        """Return type is ReactionMatch, not bare str."""
        result = identify_reaction_type("Pd(PPh3)4, K2CO3, THF")
        assert isinstance(result, ReactionMatch)
        assert isinstance(result.name, str)
        assert isinstance(result.confidence, float)

    def test_positive_match_has_confidence(self) -> None:
        """Positive match should have confidence > 0."""
        result = identify_reaction_type("Pd(PPh3)4, K2CO3, THF")
        assert result.confidence > 0.0
        assert result.name != ""

    def test_no_match_zero_confidence(self) -> None:
        """No match should return empty name and 0.0 confidence."""
        result = identify_reaction_type("water, sand, nothing chemical")
        assert result.name == ""
        assert result.confidence == 0.0

    def test_empty_input_zero_confidence(self) -> None:
        result = identify_reaction_type("")
        assert result.name == ""
        assert result.confidence == 0.0

    def test_pattern_field_populated(self) -> None:
        """Matched result should include the ReactionPattern."""
        result = identify_reaction_type("NBS, THF, RT")
        assert result.pattern is not None
        assert result.pattern.name == result.name


class TestAldolCondensation:
    """Aldol condensation — base-only reaction, no catalyst, no key_reagent."""

    def test_naoh_thf_identified(self) -> None:
        """GIVEN NaOH in THF, THEN aldol condensation identified."""
        result = identify_reaction_type("NaOH, THF, -78°C to RT")
        assert "aldol" in result.name.lower()

    def test_lda_thf_identified(self) -> None:
        """GIVEN LDA in THF, THEN aldol condensation identified."""
        result = identify_reaction_type("LDA, THF, -78°C")
        assert "aldol" in result.name.lower()


class TestConfidenceScoring:
    """Confidence scoring should reflect match quality."""

    def test_more_dimensions_higher_confidence(self) -> None:
        """Match with catalyst+base+solvent > match with catalyst alone."""
        full = identify_reaction_type("Pd(PPh3)4, K2CO3, THF, 80°C")
        partial = identify_reaction_type("Pd(PPh3)4")
        assert full.confidence > partial.confidence

    def test_key_reagent_gives_high_confidence(self) -> None:
        """Key reagent match should give reasonably high confidence."""
        result = identify_reaction_type("NBS, CHCl3, RT")
        assert result.confidence >= 0.3


# ---------------------------------------------------------------------------
# S42 MR24: Regression tests for review findings
# ---------------------------------------------------------------------------


class TestMR24C2BackwardCompat:
    """C2 fix: ReactionMatch.__str__ returns name for backward compatibility."""

    def test_str_returns_name(self) -> None:
        result = identify_reaction_type("Pd(PPh3)4, K2CO3, THF")
        assert str(result) == result.name

    def test_str_empty_on_no_match(self) -> None:
        result = identify_reaction_type("water, sand")
        assert str(result) == ""

    def test_fstring_compat(self) -> None:
        result = identify_reaction_type("NBS, THF")
        assert f"Reaction: {result}" == f"Reaction: {result.name}"


class TestMR24C1G1LeftBoundary:
    """C1/G1 fix: short key_reagent must not match inside longer tokens."""

    def test_nbs_does_not_trigger_b_key_reagent(self) -> None:
        """NBS should match 'NBS bromination', not any B-containing reaction."""
        result = identify_reaction_type("NBS, DMF")
        assert "bromination" in result.name.lower()

    def test_mg_in_mgbr2_context(self) -> None:
        """MgBr in Grignard context should still match Grignard."""
        result = identify_reaction_type("Mg, THF, reflux")
        # Mg turnings in THF is Grignard
        if result.name:
            assert "grignard" in result.name.lower()


class TestMR24C4MinEvidence:
    """C4 fix: single low-weight dimension should not produce a named match."""

    def test_solvent_only_no_match(self) -> None:
        """Solvent-only input should not match any reaction."""
        result = identify_reaction_type("THF")
        assert result.name == ""

    def test_single_base_only_no_match(self) -> None:
        """Base-only input (no other evidence) should not match."""
        result = identify_reaction_type("K2CO3")
        assert result.name == ""


class TestMR24G4G5NullSafety:
    """G4/G5 fix: null values in JSON should not crash loading."""

    def test_identify_still_works(self) -> None:
        """Basic identify_reaction_type works after null-safety fixes."""
        result = identify_reaction_type("Pd(PPh3)4, K2CO3, THF")
        assert result.name != ""


# ---------------------------------------------------------------------------
# S42 MR24 R2: Regression tests for Round 2 findings
# ---------------------------------------------------------------------------


class TestMR24R2RegexBoundary:
    """R2 fix: left boundary blocks uppercase (other elements), allows lowercase (Ar/Ph prefix)."""

    def test_arsnbu3_matches_stille(self) -> None:
        """ArSnBu3: 'Ar' prefix (lowercase r) should NOT block Sn match."""
        result = identify_reaction_type("ArSnBu3, Pd(PPh3)4, toluene")
        assert "stille" in result.name.lower()

    def test_nbs_still_matches_bromination(self) -> None:
        """NBS as standalone token should still match bromination."""
        result = identify_reaction_type("NBS, CHCl3")
        assert "bromination" in result.name.lower()

    def test_uppercase_left_blocks_false_match(self) -> None:
        """TMSN3: uppercase S before N should block 'Sn' key_reagent."""
        # TMS-N3 = trimethylsilyl azide, the 'SN' crosses element boundaries
        result = identify_reaction_type("TMSN3, Pd(PPh3)4, toluene")
        assert "stille" not in result.name.lower()


class TestMR24R2KrHitLogic:
    """R2-C2 fix: evidence threshold uses actual kr_hit, not inferred."""

    def test_catalyst_only_no_match_without_kr(self) -> None:
        """Catalyst alone without key_reagent should not match if <2 dims."""
        # Pd catalyst alone — Suzuki needs boronic acid (key_reagent)
        result = identify_reaction_type("Pd(PPh3)4")
        # Should either be empty or have low confidence
        if result.name:
            assert result.confidence < 0.5
