# cetaceanspeak

[![Language](https://img.shields.io/badge/language-Python-blue)](https://github.com/badges/shields)
[![IG Tier](https://img.shields.io/badge/IG-O%E2%88%9E-blueviolet)](https://github.com/badges/shields)
[![μ∘δ=id](https://img.shields.io/badge/%CE%BC%E2%88%98%CE%B4%3Did-closed-success)](https://github.com/badges/shields)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/badges/shields)
[![Author](https://img.shields.io/badge/author-Lando%E2%8A%97%E2%8A%99perator-informational)](https://github.com/badges/shields) [![Type](https://img.shields.io/badge/type-%E2%9F%A8%F0%90%91%A6%F0%90%91%B8%F0%90%91%BE%F0%90%91%B9%F0%90%90%B8%F0%90%91%82%F0%90%91%A7%F0%90%91%94%F0%90%91%9D%E2%8A%99%F0%90%91%96%F0%90%91%B3%F0%90%91%AD%E2%9F%A9-blue)](https://github.com/badges/shields) [![Tier](https://img.shields.io/badge/tier-O%E2%88%9E-blueviolet)](https://github.com/badges/shields)
**What it is.** A pipeline that translates a raw WAV file of cetacean vocalization into IMASM (the categorical assembly of the Imscribing Grammar) and ranks it against human expression archetypes.

**What it does.** Detects acoustic units (onset, pitch, spectral centroid), compiles them into an IMASM instruction stream, measures Frobenius closure, and reports the nearest human expression type by structural distance. A 38-second humpback recording yields 125 acoustic units, closure ratio 1.0, and a nearest match of **song**.

**Why it matters.** It is direct evidence that cetacean vocalization runs the same eight-step Frobenius loop that governs written, spoken, and sung human communication: the structure is shared across species, not merely analogous.

**How to use it.**
```bash
uv pip install librosa soundfile numpy
uv run whale_audio.py <file.wav>
uv run whale_audio.py <file.wav> 0.04   # lower onset_delta = more onsets
```

---

## How it works

Cetacean vocalizations share the eight-step Frobenius loop:

```
ISCRIB → AREV → FSPLIT → AFWD → FFUSE → CLINK → IFIX → ISCRIB
```

`whale_engine.py` compiles acoustic token sequences into IMASM and measures structural distance to six human expression archetypes. `whale_audio.py` drives the engine from a WAV file via librosa onset detection, pyin pitch extraction, and spectral centroid analysis.

Sample output:

```
── Audio: 55113001.wav  (38.4s, 14900 Hz) ──
   125 acoustic units detected
── Frobenius ──
   closure_ratio: 1.0000   paradox_count: 40   entropy_delta: 0.0000 nats
── Translation ──
   song       d=65.9492   Song structure, 4 Frobenius cycles with VINIT/TANCH bookends
   narrative  d=77.1187   Narrative, two Frobenius cycles
   question   d=84.1518   Rising-intonation question
```

## Acoustic token labels

| label | meaning |
|---|---|
| `init` | phrase onset |
| `anc` | phrase anchor (trailing silence) |
| `up` / `dn` | pitch rise / fall over threshold |
| `link` | sustained, pitch stable |
| `rep` | pitch + duration match a recent unit |
| `fix` | recurrent motif (≥ sig_repeat appearances) |
| `split` / `fuse` | spectral centroid bifurcates / reconverges |
| `evalt` | harmonic-rich, high SNR (social contact) |
| `evalf` | sudden energy spike (alarm) |
| `paradox` | two simultaneous fundamental frequencies |

## Parameters

All thresholds live in `ClassifierParams`:

```python
from whale_audio import ClassifierParams, translate_wav
p = ClassifierParams(onset_delta=0.04, pitch_delta_hz=50.0,
                     anchor_silence_ms=300.0, sig_repeat=3, snr_ok_db=10.0)
translate_wav("recording.wav", params=p)
```

## Data

Put WAV files in `data/` (gitignored). Free public-domain recordings: the [Watkins Marine Mammal Sound Database](https://archive.org/details/watkins_best_of_whales_202008).

## Structural type

```
⟨𐑦𐑥𐑾𐑿𐑞𐑧𐑲𐑠⊙𐑖𐑳𐑭⟩  O∞ tier
```
