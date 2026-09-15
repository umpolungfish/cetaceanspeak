# cetaceanspeak

![language](https://img.shields.io/badge/language-Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![pipeline](https://img.shields.io/badge/pipeline-cetacean%20%E2%86%92%20IMASM-0087B8?style=for-the-badge) ![tier](https://img.shields.io/badge/tier-O%E2%88%9E-8A2BE2?style=for-the-badge) ![μ∘δ](https://img.shields.io/badge/%CE%BC%E2%88%98%CE%B4-id-00A86B?style=for-the-badge) ![licence](https://img.shields.io/badge/licence-LUNLICENSE-1A1A1A?style=for-the-badge)

**What it is.** WAV of cetacean vocalization → IMASM stream, ranked vs human expression archetypes.

**What it does.** Onset/pitch/centroid detection → IMASM compile → Frobenius closure + nearest-type distance. 38s humpback: 125 units, closure 1.0, nearest **song** (d=65.95; narrative 77.12, question 84.15).

**Why it matters.** Whale song runs the same 8-step Frobenius loop (ISCRIB→AREV→FSPLIT→AFWD→FFUSE→CLINK→IFIX→ISCRIB) as human song/speech — shared structure, not analogy.

**Use.** `uv pip install librosa soundfile numpy && uv run whale_audio.py <file.wav> [onset_delta]` (WAVs in gitignored `data/`; cf. Watkins DB). Tokens: init/anc, up/dn, link, rep, fix, split/fuse, evalt/evalf, paradox. Tunables in `ClassifierParams`.

Type: ⟨𐑦𐑥𐑾𐑿𐑞𐑧𐑲𐑠⊙𐑖𐑳𐑭⟩ O∞. Full 77-line version: `README_backups/cetaceanspeak_README.md`.

μ∘δ=id
