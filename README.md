# cetaceanspeak

WAV file → cetacean vocalization translation via the IMASM compiler pipeline.

A 38-second humpback recording produces 125 acoustic units, Frobenius closure ratio 1.0, and a ranked match against six human expression archetypes. The closest match on the Watkins database sample is **song** (d=65.95).

## How it works

Cetacean vocalizations share the same eight-step Frobenius loop that governs written, spoken, and sung human communication:

```
ISCRIB → AREV → FSPLIT → AFWD → FFUSE → CLINK → IFIX → ISCRIB
```

`whale_engine.py` compiles acoustic token sequences into IMASM instruction streams and measures structural distance to six human expression archetypes. `whale_audio.py` drives that engine from a raw WAV file using librosa onset detection, pyin pitch extraction, and spectral centroid analysis.

## Usage

```bash
uv pip install librosa soundfile numpy
uv run whale_audio.py <file.wav>
uv run whale_audio.py <file.wav> 0.04   # lower onset_delta = more onsets
```

Output:

```
── Audio: 55113001.wav  (38.4s, 14900 Hz) ──
   125 acoustic units detected

      68.7 ms  init            52.5 Hz   -6.5 dB
     859.1 ms  up             528.9 Hz   -5.5 dB
    ...

── Frobenius ──
   closure_ratio: 1.0000
   paradox_count: 40
   entropy_delta: 0.0000 nats

── Translation ──
   song                      d=65.9492  — Song structure — 4 Frobenius cycles with VINIT/TANCH bookends
   narrative                 d=77.1187  — Narrative with two Frobenius cycles — the eight-instruction loop ×2
   question                  d=84.1518  — Rising intonation question — two upward steps, one down, linked
```

## Acoustic token labels

| label | meaning |
|---|---|
| `init` | phrase onset |
| `anc` | phrase anchor (trailing silence) |
| `up` | pitch rise > threshold |
| `dn` | pitch fall > threshold |
| `link` | sustained, pitch stable |
| `rep` | pitch + duration match a recent unit |
| `fix` | recurrent motif (≥ sig_repeat appearances) |
| `split` | spectral centroid bifurcates |
| `fuse` | centroid converges after split |
| `evalt` | harmonic-rich, high SNR — social contact |
| `evalf` | sudden energy spike — alarm |
| `paradox` | two simultaneous fundamental frequencies |

## Parameters

All thresholds are in `ClassifierParams`:

```python
from whale_audio import ClassifierParams, translate_wav

p = ClassifierParams(
    onset_delta=0.04,       # lower = more onsets detected
    pitch_delta_hz=50.0,    # min Hz change to call rise/fall
    anchor_silence_ms=300.0,
    sig_repeat=3,           # appearances before → fix token
    snr_ok_db=10.0,
)
translate_wav("recording.wav", params=p)
```

## Data

Put WAV files in `data/` (gitignored). Free public domain recordings:
- [Watkins Marine Mammal Sound Database](https://archive.org/details/watkins_best_of_whales_202008) — humpback, sperm, blue, orca, and more

## Structural type

```
⟨𐑦·𐑥·𐑾·𐑿·𐑞·𐑧·𐑲·𐑠·⊙·𐑖·𐑳·𐑭⟩  O_∞ tier
```
