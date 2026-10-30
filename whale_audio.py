"""
whale_audio.py — Audio ingestion layer for whale_engine.py

WAV file → (timestamps, labels, frequencies) → whale_engine.WhaleCompiler.from_spectrogram()

Classification rules (heuristic; tunable):
  init      — first onset after silence (energy below floor)
  anc       — last onset before silence (trailing silence > anchor_silence_ms)
  up        — pitch rises > pitch_delta_hz between this and previous unit
  dn        — pitch falls > pitch_delta_hz
  link      — sustained energy, pitch stable (delta < pitch_delta_hz)
  rep       — pitch and duration within rep_tolerance of any recent unit
  split     — spectral centroid bifurcates: two distinct peaks detected
  fuse      — spectral centroid converges after a split
  evalt     — harmonic-rich, low noise floor (SNR > snr_ok_db), no alarm signature
  evalf     — sudden energy spike (> alarm_spike_db above rolling mean)
  paradox   — two simultaneous fundamental frequencies (overlapping calls)
  fix       — unit matches a recurrent motif that has appeared >= sig_repeat times
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from collections import Counter

try:
    import librosa
    import soundfile as sf
    _HAS_AUDIO = True
except ImportError:
    _HAS_AUDIO = False


# ── tunable parameters ────────────────────────────────────────────────────────

@dataclass
class ClassifierParams:
    pitch_delta_hz:      float = 50.0    # min Hz change to call rise/fall
    anchor_silence_ms:   float = 300.0   # trailing silence → anchor token
    rep_tolerance:       float = 0.15    # 15 % pitch/duration tolerance for repeat
    sig_repeat:          int   = 3       # appearances before → signature token
    alarm_spike_db:      float = 12.0   # dB above rolling mean → alarm
    snr_ok_db:           float = 10.0   # SNR threshold for social_ok
    onset_delta:         float = 0.07   # librosa onset sensitivity (lower = more)
    frame_ms:            float = 23.2   # hop length converted to ms (512/sr * 1000)
    split_peak_ratio:    float = 0.4    # secondary peak / primary peak → split
    fmin:                float = 0.0    # pyin fmin override (0 = auto from sr)
    fmax:                float = 0.0    # pyin fmax override (0 = auto from sr)


# ── result container ──────────────────────────────────────────────────────────

@dataclass
class AudioUnit:
    timestamp_ms: float
    duration_ms:  float
    freq_hz:      float
    label:        str
    energy_db:    float = 0.0
    snr_db:       float = 0.0
    notes:        list[str] = field(default_factory=list)


# ── core pipeline ─────────────────────────────────────────────────────────────

def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    if not _HAS_AUDIO:
        raise ImportError("uv pip install librosa soundfile")
    y, sr = librosa.load(str(path), sr=None, mono=True)
    return y, sr


def classify_audio(
    y: np.ndarray,
    sr: int,
    params: Optional[ClassifierParams] = None,
) -> list[AudioUnit]:
    """Segment a mono audio array into labelled acoustic units."""
    if params is None:
        params = ClassifierParams()

    hop = 512
    params.frame_ms = hop / sr * 1000

    # ── onset detection ───────────────────────────────────────────────────────
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=hop,
        delta=params.onset_delta, backtrack=True,
    )
    if len(onset_frames) == 0:
        return []
    onset_times_ms = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop) * 1000

    # ── per-frame pitch (pyin is better but slower; use yin) ─────────────────
    _fmin = params.fmin if params.fmin > 0 else max(20.0, sr / 2048 * 1.1)
    _fmax = params.fmax if params.fmax > 0 else min(8000.0, sr / 2 - 1)
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=_fmin, fmax=_fmax, sr=sr, hop_length=hop,
        fill_na=0.0,
    )

    # ── per-frame energy ──────────────────────────────────────────────────────
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_db = librosa.amplitude_to_db(rms + 1e-9, ref=np.max)

    # ── spectral centroids (for split detection) ──────────────────────────────
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]

    # ── rolling noise floor ───────────────────────────────────────────────────
    win = max(1, len(rms_db) // 10)
    noise_floor = np.convolve(rms_db, np.ones(win) / win, mode='same')

    def frame_at(ms: float) -> int:
        return min(int(ms / params.frame_ms), len(f0) - 1)

    def pitch_at(ms: float) -> float:
        f = frame_at(ms)
        return float(f0[f]) if f0[f] > 0 else 0.0

    def energy_at(ms: float) -> float:
        return float(rms_db[frame_at(ms)])

    # ── build units ───────────────────────────────────────────────────────────
    units: list[AudioUnit] = []
    prev_pitch: float = 0.0
    recent_pitches: list[float] = []
    recent_durations: list[float] = []
    motif_counter: Counter = Counter()
    in_split = False

    total_ms = len(y) / sr * 1000

    for i, t_ms in enumerate(onset_times_ms):
        next_ms = onset_times_ms[i + 1] if i + 1 < len(onset_times_ms) else total_ms
        duration_ms = next_ms - t_ms
        pitch = pitch_at(t_ms)
        energy = energy_at(t_ms)
        noise = float(noise_floor[frame_at(t_ms)])
        snr = energy - noise
        notes: list[str] = []

        # silence check for init / anc
        is_first = (i == 0)
        is_last = (i == len(onset_times_ms) - 1)
        trailing_silence = total_ms - next_ms if is_last else 0.0

        # spectral bifurcation (split detection)
        f_start = frame_at(t_ms)
        f_end = frame_at(next_ms)
        seg_cent = cent[f_start:f_end] if f_end > f_start else cent[f_start:f_start+1]
        cent_std = float(np.std(seg_cent)) if len(seg_cent) > 1 else 0.0
        has_split = cent_std > (float(np.mean(seg_cent)) * params.split_peak_ratio)

        # motif fingerprint: quantise pitch + duration to nearest 50 Hz / 50 ms
        motif_key = (round(pitch / 50) * 50, round(duration_ms / 50) * 50)
        motif_counter[motif_key] += 1
        is_signature = motif_counter[motif_key] >= params.sig_repeat

        # sudden energy spike → alarm
        is_alarm = (energy - noise) > params.alarm_spike_db and energy > -20

        # simultaneous f0 (two voices) — check neighbouring bins via harmonic product
        f_mid = (f_start + f_end) // 2
        spec = np.abs(librosa.stft(y[max(0, f_mid*hop - hop*4): f_mid*hop + hop*4],
                                   hop_length=hop)[:, 0])
        peaks = _local_peaks(spec)
        is_paradox = len(peaks) >= 2 and peaks[1] / (peaks[0] + 1e-9) > 0.5

        # repeat: pitch & duration within tolerance of any recent unit
        is_rep = any(
            abs(p - pitch) / (max(p, pitch, 1)) < params.rep_tolerance and
            abs(d - duration_ms) / (max(d, duration_ms, 1)) < params.rep_tolerance
            for p, d in zip(recent_pitches[-6:], recent_durations[-6:])
        )

        # pitch direction
        pitch_delta = pitch - prev_pitch if prev_pitch > 0 and pitch > 0 else 0.0
        is_rise = pitch_delta >  params.pitch_delta_hz
        is_fall = pitch_delta < -params.pitch_delta_hz
        is_stable = abs(pitch_delta) <= params.pitch_delta_hz

        # classify — priority order matters
        if is_first:
            label = "init"
        elif is_last and trailing_silence > params.anchor_silence_ms:
            label = "anc"
        elif is_signature:
            label = "fix"
        elif is_paradox:
            label = "paradox"
            notes.append("simultaneous_f0")
        elif is_alarm:
            label = "evalf"
        elif has_split and not in_split:
            label = "split"
            in_split = True
        elif in_split and is_stable:
            label = "fuse"
            in_split = False
        elif is_rep:
            label = "rep"
        elif is_rise:
            label = "up"
        elif is_fall:
            label = "dn"
        elif snr > params.snr_ok_db and not is_alarm:
            label = "evalt"
        else:
            label = "link"

        units.append(AudioUnit(
            timestamp_ms=t_ms,
            duration_ms=duration_ms,
            freq_hz=pitch,
            label=label,
            energy_db=energy,
            snr_db=snr,
            notes=notes,
        ))

        prev_pitch = pitch if pitch > 0 else prev_pitch
        recent_pitches.append(pitch)
        recent_durations.append(duration_ms)

    return units


def _local_peaks(spec: np.ndarray, min_gap: int = 5) -> list[float]:
    """Return magnitudes of local maxima in a spectrum, descending."""
    peaks = []
    for i in range(1, len(spec) - 1):
        if spec[i] > spec[i-1] and spec[i] > spec[i+1]:
            peaks.append(float(spec[i]))
    return sorted(peaks, reverse=True)


# ── Sperm whale coda classifier ───────────────────────────────────────────────
# Sperm whale codas are click trains. Each WAV in the DSWP dataset IS one coda.
# Classification is based on spectral centroid (vowel quality) and ICI pattern.
#
# a-coda: dominant energy < A_I_BOUNDARY_HZ → "dn"  (Beguš: low resonance, longer)
# i-coda: dominant energy > A_I_BOUNDARY_HZ → "up"  (Beguš: high resonance, shorter)
# ī-coda: i-coda + duration > I_LONG_MS     → "up" + "evalt"
# diphthong: centroid sweeps > DIPH_SLOPE_HZ_PER_S → "up" + "split" + "dn" + "fuse"
# signature: matches recurring ICI pattern   → "fix"
# overlap:   multiple simultaneous click trains → "paradox"

_A_I_BOUNDARY_HZ  = 1500.0   # centroid below = a-coda, above = i-coda
_I_LONG_MS        = 1200.0   # duration above this → ī-coda
_DIPH_SLOPE_THRESH = 400.0   # |Hz/s| centroid slope → diphthong
_MAX_ICI_MS       = 600.0    # clicks further apart than this = separate codas


def _detect_clicks(y: np.ndarray, sr: int,
                   onset_delta: float = 0.01) -> np.ndarray:
    """Return onset times in seconds for individual clicks."""
    hop = 256
    frames = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=hop, delta=onset_delta, backtrack=True,
    )
    return librosa.frames_to_time(frames, sr=sr, hop_length=hop)


def classify_sperm_whale_audio(
    y: np.ndarray,
    sr: int,
    params: Optional[ClassifierParams] = None,
) -> list[AudioUnit]:
    """Classify a sperm whale coda (one WAV = one coda, DSWP format).

    Classification is ICI-based, not carrier-frequency-based.  Sperm whale
    clicks are broadband (5-20 kHz), so spectral centroid does not distinguish
    vowel quality.  Vowel character lives in the click-rate rhythm:

      Regular, fast ICI  (<= 65ms mean, CV < 0.4) → i-coda  ("up")
      Regular, slow ICI  (>= 85ms mean, CV < 0.4) → a-coda  ("dn")
      Rubato  (CV 0.4-0.5, moderate slope)         → diphthong ("up/dn split fuse")
      Highly irregular (CV >= 0.5) or true overlap → paradox ("paradox")
      Long i-coda (i + dur > I_LONG_MS)            → ī-coda ("up evalt")

    Every coda ends with "fix anc" — codas are always stereotyped motifs.
    """
    if params is None:
        params = ClassifierParams()
    total_ms = len(y) / sr * 1000.0

    # ── click detection (fine-grained onset) ─────────────────────────────────
    hop_click = 64
    frames = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=hop_click, delta=0.005, backtrack=True,
    )
    click_times_s = librosa.frames_to_time(frames, sr=sr, hop_length=hop_click)

    if len(click_times_s) > 1:
        icis_ms = np.diff(click_times_s) * 1000.0
        icis_ms = icis_ms[(icis_ms > 2.0) & (icis_ms < _MAX_ICI_MS)]
    else:
        icis_ms = np.array([])

    n_clicks  = len(click_times_s)
    mean_ici  = float(np.mean(icis_ms))  if len(icis_ms) else 0.0
    cv_ici    = float(np.std(icis_ms) / (mean_ici + 1e-9)) if len(icis_ms) else 0.0
    ici_slope = float(np.polyfit(range(len(icis_ms)), icis_ms, 1)[0]) \
                if len(icis_ms) > 3 else 0.0  # ms/click; + = slowing, - = speeding

    # ── simultaneous click trains (two animals) ───────────────────────────────
    # Bimodal ICI distribution: if ICI has two well-separated clusters
    if len(icis_ms) > 6:
        short_frac = float(np.mean(icis_ms < 40.0))
        long_frac  = float(np.mean(icis_ms > 120.0))
        two_trains = short_frac > 0.25 and long_frac > 0.25
    else:
        two_trains = False

    # ── classify ─────────────────────────────────────────────────────────────
    labels: list[str] = ["init"]
    notes = [f"n_clicks={n_clicks}",
             f"mean_ICI={mean_ici:.1f}ms",
             f"cv={cv_ici:.3f}",
             f"slope={ici_slope:.2f}ms/click"]

    if two_trains:
        # Two animals clicking simultaneously — genuine paradox
        labels.append("paradox")
    elif cv_ici >= 0.5:
        # Highly irregular rubato: rhythm is dialetheic (both fast and slow)
        labels.append("paradox")
    elif cv_ici >= 0.4:
        # Moderate rubato — diphthong: rhythm sweeps across a/i
        if ici_slope < -0.5:
            labels += ["dn", "split", "up", "fuse"]   # slowing→speeding: a→i
        elif ici_slope > 0.5:
            labels += ["up", "split", "dn", "fuse"]   # speeding→slowing: i→a
        else:
            labels.append("paradox")                   # rubato without clear sweep
    elif mean_ici <= 65.0:
        # Fast regular clicks → i-coda
        if total_ms > _I_LONG_MS:
            labels += ["up", "evalt"]                  # ī-coda: sustained
        else:
            labels.append("up")
    elif mean_ici >= 85.0:
        # Slow regular clicks → a-coda
        labels.append("dn")
    else:
        # Borderline (65-85ms): use slope to break tie
        if ici_slope < -0.3:
            labels += ["dn", "split", "up", "fuse"]   # decelerating: a→i diphthong
        elif ici_slope > 0.3:
            labels += ["up", "split", "dn", "fuse"]   # accelerating: i→a diphthong
        else:
            labels.append("link")                      # neutral carrier

    labels += ["fix", "anc"]

    # ── build AudioUnit list ─────────────────────────────────────────────────
    step_ms = total_ms / len(labels)
    return [
        AudioUnit(
            timestamp_ms=i * step_ms,
            duration_ms=step_ms,
            freq_hz=mean_ici,          # ICI stored in freq_hz for display
            label=lbl,
            energy_db=float(n_clicks),
            snr_db=cv_ici,
            notes=notes,
        )
        for i, lbl in enumerate(labels)
    ]


# ── public entry point ────────────────────────────────────────────────────────

def translate_wav(
    path: str | Path,
    params: Optional[ClassifierParams] = None,
    verbose: bool = True,
    species: str = "auto",
) -> dict:
    """Full pipeline: WAV → acoustic units → IMASM → Frobenius → translation.

    Args:
        species: "humpback" | "orca" | "sperm_whale" | "auto" (default).
                 "sperm_whale" uses the coda-level classifier.
                 "auto" uses the general onset classifier.

    Returns the whale_engine.StructuralAlignment result dict.
    """
    from whale_engine import WhaleCompiler, StructuralAlignment, FrobeniusAnalyzer

    y, sr = load_wav(path)
    if species == "sperm_whale":
        units = classify_sperm_whale_audio(y, sr, params)
    else:
        units = classify_audio(y, sr, params)

    if not units:
        raise ValueError("No onsets detected — check file or lower onset_delta")

    timestamps = [u.timestamp_ms for u in units]
    labels     = [u.label        for u in units]
    frequencies = [u.freq_hz     for u in units]

    if verbose:
        print(f"\n── Audio: {Path(path).name}  ({len(y)/sr:.1f}s, {sr} Hz) ──")
        print(f"   {len(units)} acoustic units detected\n")
        for u in units:
            note = f"  [{', '.join(u.notes)}]" if u.notes else ""
            print(f"   {u.timestamp_ms:7.1f} ms  {u.label:<12}  "
                  f"{u.freq_hz:6.1f} Hz  {u.energy_db:5.1f} dB{note}")

    instrs = WhaleCompiler.from_spectrogram(timestamps, labels, frequencies)
    report = FrobeniusAnalyzer.analyze(instrs)
    whale_sig = FrobeniusAnalyzer.structural_signature(instrs)
    ranked = StructuralAlignment.translate(whale_sig)

    if verbose:
        print(f"\n── Frobenius ──")
        print(f"   closure_ratio: {report.closure_ratio:.4f}")
        print(f"   paradox_count: {report.paradox_count}")
        print(f"   entropy_delta: {report.entropy_delta:.4f} nats")
        print(f"\n── Translation ──")
        for expr, dist, desc in ranked:
            print(f"   {expr:<24}  d={dist:.4f}  — {desc}")

    return {"units": units, "report": report, "ranked": ranked}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import sys
    if len(sys.argv) < 2:
        print("usage: cetacean-speak <file.wav> [onset_delta]")
        print("       onset_delta: float 0.01–0.2, default 0.07 (lower = more onsets)")
        sys.exit(0)

    p = ClassifierParams()
    if len(sys.argv) >= 3:
        p.onset_delta = float(sys.argv[2])

    translate_wav(sys.argv[1], params=p)


if __name__ == "__main__":
    main()
