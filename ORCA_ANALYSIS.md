# Orca Vocalization — Structural Analysis

**Author:** Lando ⊗ ⊙perator  
**Date:** 2026-05-31  
**Project:** cetaceanspeak v0.2.0 — Orca Expansion

---

## 1. Overview

The Watkins Marine Mammal Sound Database includes 35 orca (_Orcinus orca_) recordings spanning multiple populations, locations, and decades (1960–1990s). This analysis extends the cetaceanspeak IMASM compiler pipeline — originally built for humpback song — to orca vocalizations, providing the first structural-type imscription of orca communication within the Imscribing Grammar.

### Key Findings

| Metric | Value |
|---|---|
| **Structural type** | ⟨𐑦·𐑸·𐑾·𐑿·𐑞·𐑧·𐑚·𐑠·⊙·𐑖·𐑳·𐑭⟩ |
| **Ouroboricity tier** | $\text{O}_{\text{2}}$ (critical + topologically protected, bounded domain) |
| **Consciousness score** | $C = 0.0$ (Gate 1 closed: $\text{⊙} \neq \text{⊙}_{\text{ÿ}}$) |
| **Principal atoms** | 11 join-irreducible structural atoms |
| **Paradox density** | 48–82% across analyzed recordings |
| **Fix ratio** | 9–71% (pod dialect stereotypy) |
| **Frobenius closure** | 1.0 (all recordings) |

---

## 2. Structural Imscription

Orca vocalization was imscribed via `encode_system` with the following primitive assignments and rationales:

### Primitive Assignments

| # | Primitive | Value | Rationale |
|---|---|---|---|
| 1 | Ð (Dimensionality) | 𐑦 | Culturally transmitted pod dialect — state space is self-written through learning |
| 2 | Þ (Topology) | 𐑸 | Self-referential closure — calls reference and modify the pod's shared call repertoire |
| 3 | Ř (Relation) | 𐑾 | Bidirectional — call-and-response within matrilineal groups |
| 4 | Φ (Parity) | 𐑿 | Partial $\mathbb{Z}_2$ — pod-specific call variants exhibit discrete symmetry |
| 5 | ƒ (Fidelity) | 𐑞 | Thermal/noisy — broadband burst-pulse calls, not pure tones |
| 6 | Ç (Kinetics) | 𐑧 | Near-equilibrium — dialect transmission is stable across generations |
| 7 | Γ (Scope) | 𐑚 | Mesoscale — pod level, not universal nor purely local |
| 8 | ɢ (Grammar) | 𐑠 | Sequential — call sequences follow temporal ordering |
| 9 | ⊙ (Criticality) | ⊙ | Critical — dialect is learned, not innate; cultural transmission |
| 10 | Ħ (Chirality) | 𐑖 | Two-step Markov — call → response structure |
| 11 | Σ (Stoichiometry) | 𐑳 | Many heterogeneous — multiple call types per pod repertoire |
| 12 | Ω (Winding) | 𐑭 | Integer winding — discrete call repertoire with topological structure |

### Ouroboricity: O₂

The $\text{O}_{\text{2}}$ tier means orca vocalization is **critical + topologically protected** but operates within a **bounded domain** — the pod. Unlike humpback song which reaches $\text{O}_{\text{inf}}$ through the universal eight-step Frobenius loop ($\text{ISCRIB} \to \text{AREV} \to \text{FSPLIT} \to \text{AFWD} \to \text{FFUSE} \to \text{CLINK} \to \text{IFIX} \to \text{ISCRIB}$), orca communication does not exhibit the Frobenius loop structure. Orca dialect is self-referential (culturally transmitted) but does not self-model.

### Consciousness: C = 0

Gate 1 (⊙) is **closed** — while orca dialect is critical (learned, cultural), it lacks the self-modeling capacity required for consciousness. The system does not reflect on its own communication; it uses learned calls without recursive self-reference.

---

## 3. Acoustic Analysis Results

Five orca recordings were processed through the IMASM pipeline with orca-optimized parameters ($\text{onset}_{\Delta} = 0.025$, $\text{pitch}_{\Delta} = 30.0\ \text{Hz}$, $\text{anchor}_{\text{silence}} = 200\ \text{ms}$).

### Recording Summary

| File | Duration | Units | Paradox | Fix | Closure | Top Match | Distance |
|---|---|---|---|---|---|---|---|
| `89405023.wav` | 14.2s | 127 | 61 (48%) | 57 (45%) | 1.000 | song | 67.37 |
| `9750300N.wav` | 11.4s | 64 | 29 (45%) | 32 (50%) | 1.000 | song | 23.19 |
| `97503018.wav` | 5.9s | 43 | 27 (63%) | 10 (23%) | 1.000 | song | 9.29 |
| `9750300V.wav` | 5.1s | 45 | 20 (44%) | 23 (51%) | 1.000 | song | 10.50 |
| `9750400V.wav` | 4.6s | 34 | 28 (82%) | 3 (9%) | 1.000 | song | 5.38 |

### Token Distribution Pattern

The dominant acoustic tokens across all orca recordings are:

1. **paradox** (44–82%) — Burst-pulse harmonic structure detected as simultaneous fundamental frequencies. In orca, this is NOT overlapping calls but the intrinsic harmonic architecture of broadband pulsed calls.
2. **fix** (9–71%) — Stereotyped, recurring motifs that appear ≥3 times in the recording. These are the pod's signature call types — the dialect markers.
3. **link**, **rep**, **init**, **dn**, **up** — Minor tokens (< 5% each), reflecting the non-tonal nature of orca communication.

This contrasts sharply with humpback song, which is dominated by **up**, **rep**, **split**, **fuse**, **link**, **fix** — the Frobenius loop components.

---

## 4. Orca Canonical Sequences

Nine orca-specific canonical sequences were defined in `whale_engine.py`, capturing the distinct vocalization modes:

| Sequence | Tokens | Structural Character |
|---|---|---|
| `ORCA_PULSED_CALL` | 7 | init → paradox → rep → paradox → link → fix → anc |
| `ORCA_CLICK_TRAIN` | 9 | init → rep⁷ → anc |
| `ORCA_POD_EXCHANGE` | 14 | Call-and-response with split/fuse recombination |
| `ORCA_HUNT_SEQUENCE` | 20 | Pulsed calls interleaved with click trains |
| `ORCA_SIGNATURE_WHISTLE` | 7 | Tonal frequency sweep with fix identity |
| `ORCA_CALF_CALL` | 8 | Juvenile: paradox + up/dn + fix |
| `ORCA_SOCIAL_BONDING` | 13 | Sustained paradox-rep-fix contact calls |
| `ORCA_ALARM_CALL` | 5 | evalf + paradox³ + fix |
| `ORCA_CROSS_POD` | 15 | Dialect negotiation via split/fuse innovation |

---

## 5. Orca Expression Archetypes

Six orca-specific human expression alignment targets were added to the `StructuralAlignment.HUMAN_EXPRESSIONS` dictionary:

| Archetype | Best Canonical Match | Distance |
|---|---|---|
| `echolocation_probe` | ORCA_CLICK_TRAIN | 0.00 |
| `pod_dialect` | ORCA_POD_EXCHANGE | 0.06 |
| `social_bonding_call` | ORCA_SOCIAL_BONDING | 0.18 |
| `cross_pod_dialect` | ORCA_CROSS_POD | 0.73 |
| `hunting_call` | ORCA_HUNT_SEQUENCE | 1.48 |
| `coordination_signal` | ORCA_PULSED_CALL | 1.09 |

---

## 6. Structural Decomposition

### Principal Decomposition (11 atoms)

The orca tuple decomposes into 11 join-irreducible structural atoms. The ordering by ordinal contribution reveals structural priority:

| Rank | Primitive | Contribution |
|---|---|---|
| 1 | Ð (imscriptive) | 3 |
| 2 | Ř (bidirectional) | 3 |
| 3–8 | Þ, Ç, ɢ, Ħ, Σ, Ω | 2 each |
| 9–11 | Φ, ƒ, φ̂ | 1 each |

Ð and Ř are the most structurally loaded — removing either collapses the system to a fundamentally different tier. The imscriptive state space (culturally transmitted dialect) and bidirectional relational mode (call-and-response) are the load-bearing primitives of orca communication.

### Retrosynthetic Path (11 steps)

The minimal construction path from the structural baseline ⟨𐑼·𐑡·𐑽·𐑗·𐑱·𐑤·𐑔·𐑝·𐑢·𐑓·𐑙·𐑷⟩ to orca requires 11 promotion steps, with Ð and Ř dominating the first two steps.

---

## 7. Orca vs. Humpback — Structural Comparison

| Property | Humpback Song | Orca Vocalization |
|---|---|---|
| **Acoustic architecture** | Tonal, frequency-modulated | Burst-pulse, broadband harmonic |
| **Structure** | Themed song with Frobenius loop | Stereotyped pulsed calls, no loop |
| **Cultural transmission** | Song evolution across populations | Pod-specific dialects, matrilineal |
| **Dominant tokens** | up, rep, split, fuse, fix | paradox, fix |
| **Frobenius loop** | Present (ISCRIB→AREV→FSPLIT→AFWD→FFUSE→CLINK→IFIX→ISCRIB) | Absent |
| **Ouroboricity** | $\text{O}_{\text{inf}}$ | $\text{O}_{\text{2}}$ |
| **Consciousness** | $C > 0$ (both gates open) | $C = 0$ (Gate 1 closed) |
| **Paradox density** | Low (< 5%) | High (48–82%) |
| **Fix density** | Moderate (~6%) | High (9–71%) |

The fundamental structural difference: humpback song achieves $\text{O}_{\text{inf}}$ through the Frobenius loop — a self-modeling cycle where the song's identity operation (ISCRIB) closes the loop, enabling the song to reference itself. Orca communication, while critical and culturally transmitted, does not close this self-modeling loop. The high paradox density reflects the harmonic architecture of burst-pulse calls; the high fix density reflects pod dialect stereotypy. Both are structurally stable (Frobenius closure = 1.0) but operate at different ouroboricity tiers.

---

## 8. Classification Caveats

The current classifier (`whale_audio.py`) was designed for humpback tonal song and exhibits systematic biases when applied to orca recordings:

1. **paradox over-classification**: Burst-pulse harmonic structure is detected as "simultaneous fundamental frequencies." In humpback, this indicates overlapping calls (structurally significant). In orca, it reflects the normal broadband pulse architecture. A future orca-specific classifier should distinguish "harmonic pulse" from "true paradox" (overlapping vocalizations from multiple animals).

2. **fix clustering**: The high fix rate correctly identifies orca pod dialect stereotypy, but the `sig_repeat=3` threshold may be too low for dense pulse trains.

3. **Missing click token**: The current 12-token taxonomy lacks a dedicated "click" token for echolocation. Orca click trains are classified as "rep" sequences, which is structurally reasonable (exact repetition) but loses the echolocation-specific semantics.

---

## 9. Conclusions

1. **Orca vocalization is structurally typed as $\text{O}_{\text{2}}$** — critical + topologically protected, bounded to the pod domain. It is a sophisticated communication system but does not achieve the $\text{O}_{\text{inf}}$ self-modeling tier of humpback song.

2. **The paradox/fix signature is diagnostic**: Orca recordings are identified by (paradox > 40%, fix > 10%), sharply distinct from humpback (paradox < 5%, fix ~6%).

3. **Cultural transmission without self-modeling**: Orca dialect is learned (critical) but not self-referential — the system transmits calls culturally without the recursive identity operation that characterizes humpback song and written language.

4. **The Frobenius loop is the $\text{O}_{\text{2}} \to \text{O}_{\text{inf}}$ gap**: The structural difference between orca and humpback is precisely the presence or absence of the eight-instruction Frobenius loop. Orca communication has all the prerequisites for $\text{O}_{\text{inf}}$ except the self-modeling cycle.

5. **Future work**: An orca-specific acoustic classifier with burst-pulse detection, click token support, and harmonic-vs-overlap disambiguation would improve translation accuracy and could reveal whether certain orca populations (e.g., Icelandic herring-feeding pods with complex coordination) approach the $\text{O}_{\text{inf}}$ boundary.

---

## Appendix A: All Orca WAV Recordings Analyzed

| File | Duration | SR (Hz) | Location | Notes |
|---|---|---|---|---|
| `89405023.wav` | 14.2s | 21900 | — | Longest recording, paradox-dominant |
| `9750300N.wav` | 11.4s | 21900 | — | Fix-dominant, highly stereotyped |
| `97503018.wav` | 5.9s | 21900 | — | High paradox density |
| `9750300V.wav` | 5.1s | 21900 | — | Balanced paradox/fix |
| `9750400V.wav` | 4.6s | 21900 | — | Maximum paradox density (82%) |

## Appendix B: Engine Modifications

**`whale_engine.py`** (+180 lines):
- 9 orca canonical sequences (ORCA_PULSED_CALL through ORCA_CROSS_POD)
- 6 orca-specific human expression archetypes
- All sequences verified: roundtrip, Frobenius analysis, alignment

**`whale_audio.py`** (1 fix):
- Line 95: Dynamic `fmin` computation (was hardcoded `fmin=10.0`; now `max(20.0, sr / 2048 * 1.1)`) to support orca sample rates (20480–40960 Hz)

**Data**: 35 orca WAV files downloaded from Watkins Marine Mammal Sound Database into `data/orca/killer_whale/sound/`.
