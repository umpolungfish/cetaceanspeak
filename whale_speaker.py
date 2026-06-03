"""
whale_speaker.py — Cetacean Response Synthesizer.

Inverse pipeline: human intent / IMASM → whale tokens → species-appropriate audio.

Two modes:
  speak(expression, species)   → WAV  — manual: pick an archetype, synthesize it
  respond(wav_path, species)   → dict — auto: analyze incoming, plan + synthesize response

Synthesis models (acoustically simplified, IMASM-structurally correct):
  humpback    — frequency sweeps + harmonic tones (200–4000 Hz)
  sperm_whale — click trains; a-coda (AREV) vs i-coda (AFWD) phonology
  orca        — burst-pulse multi-harmonic calls

For field use with real animals, replace synthesizer output with
concatenative synthesis from Watkins/CETI recordings.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf


# ── Species ───────────────────────────────────────────────────────────────────

class Species(str, Enum):
    HUMPBACK    = "humpback"
    SPERM_WHALE = "sperm_whale"
    ORCA        = "orca"


# ── Synthesis parameters per species ─────────────────────────────────────────

@dataclass
class SynthParams:
    sample_rate:     int   = 22050
    unit_duration_s: float = 0.40    # default unit length
    f_low:           float = 300.0   # Hz — baseline / a-coda resonance
    f_mid:           float = 800.0   # Hz — mid / identity tone
    f_high:          float = 2500.0  # Hz — high / i-coda resonance
    f_alarm:         float = 3500.0  # Hz — alarm burst frequency
    gap_s:           float = 0.04    # silence between units (0 = CLINK)
    click_ici_a:     float = 0.10    # sperm whale a-coda ICI (seconds)
    click_ici_i:     float = 0.06    # sperm whale i-coda ICI
    click_n_a:       int   = 5       # a-coda clicks per coda
    click_n_i:       int   = 7       # i-coda clicks per coda
    click_dur_s:     float = 0.005   # single click length
    vibrato_rate:    float = 5.0     # Hz — humpback vibrato
    vibrato_depth:   float = 0.02    # semitones


PARAMS: dict[Species, SynthParams] = {
    Species.HUMPBACK: SynthParams(
        f_low=280.0, f_mid=700.0, f_high=1800.0, f_alarm=3000.0,
        unit_duration_s=0.45, gap_s=0.03, vibrato_rate=5.5, vibrato_depth=0.025,
    ),
    Species.SPERM_WHALE: SynthParams(
        f_low=800.0, f_mid=1200.0, f_high=2500.0, f_alarm=4000.0,
        unit_duration_s=0.55, gap_s=0.05,
        click_ici_a=0.10, click_ici_i=0.055,
        click_n_a=5, click_n_i=7, click_dur_s=0.006,
    ),
    Species.ORCA: SynthParams(
        f_low=600.0, f_mid=1500.0, f_high=4000.0, f_alarm=5000.0,
        unit_duration_s=0.20, gap_s=0.02, vibrato_rate=4.0, vibrato_depth=0.015,
    ),
}


# ── Low-level waveform primitives ─────────────────────────────────────────────

def _envelope(n: int, attack: float = 0.08, release: float = 0.12) -> np.ndarray:
    env = np.ones(n)
    a = max(1, int(n * attack))
    r = max(1, int(n * release))
    env[:a] = np.linspace(0.0, 1.0, a)
    env[n - r:] = np.linspace(1.0, 0.0, r)
    return env


def _sine(freq: float, duration_s: float, sr: int, amplitude: float = 0.6,
          phase_offset: float = 0.0) -> np.ndarray:
    n = int(sr * duration_s)
    t = np.arange(n) / sr
    wave = amplitude * np.sin(2 * math.pi * freq * t + phase_offset)
    return wave * _envelope(n)


def _sweep(f_start: float, f_end: float, duration_s: float, sr: int,
           amplitude: float = 0.6) -> np.ndarray:
    n = int(sr * duration_s)
    t = np.arange(n) / sr
    # Exponential sweep for more natural feel
    if f_start <= 0 or f_end <= 0:
        f_start = max(f_start, 1.0)
        f_end = max(f_end, 1.0)
    k = math.log(f_end / f_start) / duration_s if f_end != f_start else 0.0
    inst_freq = f_start * np.exp(k * t)
    phase = 2 * math.pi * np.cumsum(inst_freq) / sr
    wave = amplitude * np.sin(phase)
    return wave * _envelope(n)


def _vibrato_sweep(f_start: float, f_end: float, duration_s: float, sr: int,
                   vibrato_rate: float, vibrato_depth: float,
                   amplitude: float = 0.6) -> np.ndarray:
    n = int(sr * duration_s)
    t = np.arange(n) / sr
    k = math.log(max(f_end, 1) / max(f_start, 1)) / duration_s if f_end != f_start else 0.0
    carrier = f_start * np.exp(k * t)
    # vibrato_depth in semitones → frequency ratio
    vib = 2 ** (vibrato_depth * np.sin(2 * math.pi * vibrato_rate * t) / 12)
    inst_freq = carrier * vib
    phase = 2 * math.pi * np.cumsum(inst_freq) / sr
    wave = amplitude * np.sin(phase)
    return wave * _envelope(n)


def _harmonic(fundamental: float, n_harmonics: int, duration_s: float,
              sr: int, amplitude: float = 0.5) -> np.ndarray:
    n = int(sr * duration_s)
    wave = np.zeros(n)
    for k in range(1, n_harmonics + 1):
        freq = fundamental * k
        if freq < sr / 2:
            wave += (amplitude / k) * _sine(freq, duration_s, sr, amplitude=1.0)
    wave = wave / (np.max(np.abs(wave)) + 1e-9) * amplitude
    return wave * _envelope(n)


def _click(freq: float, duration_s: float, sr: int, amplitude: float = 0.7) -> np.ndarray:
    n = int(sr * duration_s)
    t = np.arange(n) / sr
    decay = np.exp(-t / (duration_s * 0.25))
    wave = amplitude * np.sin(2 * math.pi * freq * t) * decay
    return wave


def _click_train(n_clicks: int, ici_s: float, resonant_hz: float,
                 click_dur_s: float, sr: int, amplitude: float = 0.65) -> np.ndarray:
    total_s = n_clicks * ici_s + click_dur_s
    n_total = int(sr * total_s)
    wave = np.zeros(n_total)
    single = _click(resonant_hz, click_dur_s, sr, amplitude)
    for i in range(n_clicks):
        start = int(i * ici_s * sr)
        end = start + len(single)
        if end <= n_total:
            wave[start:end] += single
    # normalize
    peak = np.max(np.abs(wave))
    if peak > 0:
        wave = wave / peak * amplitude
    return wave


def _burst_pulse(fundamental: float, n_harmonics: int, duration_s: float,
                 sr: int, amplitude: float = 0.65) -> np.ndarray:
    """Orca-style burst pulse: dense multi-harmonic energy burst."""
    n = int(sr * duration_s)
    wave = np.zeros(n)
    t = np.arange(n) / sr
    for k in range(1, n_harmonics + 1):
        freq = fundamental * k
        if freq < sr / 2:
            a = amplitude / (k ** 0.6)
            phase = 2 * math.pi * freq * t + (k * 0.4)  # phase offset per harmonic
            wave += a * np.sin(phase)
    # rapid onset envelope
    env = _envelope(n, attack=0.04, release=0.15)
    wave = wave * env
    peak = np.max(np.abs(wave))
    if peak > 0:
        wave = wave / peak * amplitude
    return wave


def _silence(duration_s: float, sr: int) -> np.ndarray:
    return np.zeros(int(sr * duration_s))


# ── CetaceanSynthesizer ───────────────────────────────────────────────────────

class CetaceanSynthesizer:
    """Synthesizes a token sequence into species-appropriate audio.

    Token labels → IMASM semantics → waveform per species.
    Each token produces one acoustic unit; units are concatenated with
    species-appropriate inter-unit gaps (CLINK = no gap).
    """

    def __init__(self, species: Species) -> None:
        self.species = species
        self.p = PARAMS[species]
        self._last_unit: Optional[np.ndarray] = None

    def _unit(self, token: str) -> np.ndarray:
        sr = self.p.sample_rate
        dur = self.p.unit_duration_s
        p = self.p

        if self.species == Species.HUMPBACK:
            w = self._humpback_unit(token, sr, dur, p)
        elif self.species == Species.SPERM_WHALE:
            w = self._sperm_unit(token, sr, dur, p)
        else:
            w = self._orca_unit(token, sr, dur, p)

        self._last_unit = w
        return w

    def _humpback_unit(self, token: str, sr: int, dur: float,
                       p: SynthParams) -> np.ndarray:
        vib = lambda fs, fe: _vibrato_sweep(fs, fe, dur, sr,
                                             p.vibrato_rate, p.vibrato_depth)
        if token == "init":
            return _sine(p.f_low, 0.10, sr, amplitude=0.5)
        elif token == "anc":
            return _sine(p.f_low, 0.10, sr, amplitude=0.4)
        elif token == "up":
            return vib(p.f_low, p.f_high)
        elif token == "dn":
            return vib(p.f_high, p.f_low)
        elif token == "rep":
            return self._last_unit.copy() if self._last_unit is not None \
                else _sine(p.f_mid, dur, sr)
        elif token == "split":
            # Bifurcation: two simultaneous tones
            a = _sine(p.f_mid, dur, sr, amplitude=0.4)
            b = _sine(p.f_mid * 2.0, dur, sr, amplitude=0.3)
            return a + b
        elif token == "fuse":
            # Convergence: harmonics blend to fundamental
            a = _sine(p.f_mid * 2.0, dur * 0.5, sr, amplitude=0.35)
            b = _sine(p.f_mid, dur * 0.5, sr, amplitude=0.5)
            return np.concatenate([a, b])
        elif token == "link":
            return _silence(0.0, sr)  # no gap — handled in concatenation
        elif token == "evalt":
            return _harmonic(p.f_mid * 0.8, 4, dur * 0.8, sr, amplitude=0.55)
        elif token == "evalf":
            return _sweep(p.f_high, p.f_alarm, 0.15, sr, amplitude=0.7)
        elif token == "paradox":
            return _harmonic(p.f_low, 6, dur, sr, amplitude=0.5)
        elif token == "fix":
            return _sine(p.f_mid, dur * 1.5, sr, amplitude=0.6)
        else:
            return _sine(p.f_mid, dur * 0.3, sr, amplitude=0.3)

    def _sperm_unit(self, token: str, sr: int, dur: float,
                    p: SynthParams) -> np.ndarray:
        a_train = lambda: _click_train(p.click_n_a, p.click_ici_a,
                                        p.f_low, p.click_dur_s, sr)
        i_train = lambda: _click_train(p.click_n_i, p.click_ici_i,
                                        p.f_high, p.click_dur_s, sr)
        if token == "init":
            return _click(p.f_mid, p.click_dur_s * 2, sr, amplitude=0.6)
        elif token == "anc":
            return _click(p.f_low, p.click_dur_s * 2, sr, amplitude=0.5)
        elif token == "dn":   # a-coda
            return a_train()
        elif token == "up":   # i-coda
            return i_train()
        elif token == "rep":
            return self._last_unit.copy() if self._last_unit is not None \
                else a_train()
        elif token == "split":
            # Two interleaved click trains (a + i simultaneously)
            ta = a_train()
            ti = i_train()
            n = max(len(ta), len(ti))
            out = np.zeros(n)
            out[:len(ta)] += ta * 0.6
            out[:len(ti)] += ti * 0.5
            return out / (np.max(np.abs(out)) + 1e-9) * 0.65
        elif token == "fuse":
            # Click rate slows to a-coda rate (convergence)
            ta = _click_train(p.click_n_i, p.click_ici_i * 1.5,
                               p.f_mid, p.click_dur_s, sr)
            return ta
        elif token == "link":
            return _silence(0.03, sr)
        elif token == "evalt":
            # Sustained long-i coda (ī-coda)
            return _click_train(p.click_n_i + 3, p.click_ici_i * 1.3,
                                 p.f_mid, p.click_dur_s * 1.5, sr)
        elif token == "evalf":
            # High-frequency alarm burst
            return _click_train(10, 0.02, p.f_alarm, p.click_dur_s, sr, amplitude=0.8)
        elif token == "paradox":
            # Dense burst-pulse (multi-coda overlap)
            ta = _click_train(6, 0.04, p.f_low, p.click_dur_s, sr, amplitude=0.5)
            ti = _click_train(6, 0.035, p.f_high, p.click_dur_s, sr, amplitude=0.45)
            n = max(len(ta), len(ti))
            out = np.zeros(n)
            out[:len(ta)] += ta
            out[:len(ti)] += ti
            return out / (np.max(np.abs(out)) + 1e-9) * 0.65
        elif token == "fix":
            # Signature coda: short-long-short ICI pattern
            ta = _click_train(2, p.click_ici_a * 0.7, p.f_mid, p.click_dur_s, sr)
            tb = _click_train(1, p.click_ici_a * 1.5, p.f_mid, p.click_dur_s, sr)
            tc = _click_train(2, p.click_ici_a * 0.7, p.f_mid, p.click_dur_s, sr)
            return np.concatenate([ta, tb, tc])
        else:
            return a_train()

    def _orca_unit(self, token: str, sr: int, dur: float,
                   p: SynthParams) -> np.ndarray:
        burst = lambda n_h=5: _burst_pulse(p.f_mid, n_h, dur, sr)
        if token == "init":
            return burst(3)[:int(sr * 0.08)]
        elif token == "anc":
            return burst(3)[:int(sr * 0.08)]
        elif token == "up":
            return _vibrato_sweep(p.f_low, p.f_high, dur, sr,
                                   p.vibrato_rate, p.vibrato_depth)
        elif token == "dn":
            return _vibrato_sweep(p.f_high, p.f_low, dur, sr,
                                   p.vibrato_rate, p.vibrato_depth)
        elif token == "rep":
            return self._last_unit.copy() if self._last_unit is not None \
                else burst()
        elif token == "paradox":
            return burst(7)
        elif token == "split":
            a = burst(4)
            b = _sine(p.f_high, dur, sr, amplitude=0.35)
            n = max(len(a), len(b))
            out = np.zeros(n)
            out[:len(a)] += a * 0.6
            out[:len(b)] += b
            return out / (np.max(np.abs(out)) + 1e-9) * 0.65
        elif token == "fuse":
            return burst(3)
        elif token == "link":
            return _silence(0.01, sr)
        elif token == "evalt":
            return _harmonic(p.f_mid * 0.7, 4, dur * 1.2, sr, amplitude=0.55)
        elif token == "evalf":
            return _burst_pulse(p.f_alarm, 4, 0.12, sr, amplitude=0.75)
        elif token == "fix":
            # Stereotyped call: fundamental + harmonics, sustained
            return _harmonic(p.f_mid, 5, dur * 1.8, sr, amplitude=0.6)
        else:
            return burst()

    def synthesize(self, tokens: list[str]) -> np.ndarray:
        """Synthesize a token sequence to a waveform. Returns float32 array."""
        sr = self.p.sample_rate
        gap = _silence(self.p.gap_s, sr)
        segments: list[np.ndarray] = []

        for i, token in enumerate(tokens):
            unit = self._unit(token)
            segments.append(unit)
            # CLINK = no gap; every other inter-unit transition gets gap
            if token != "link" and i < len(tokens) - 1 and tokens[i + 1] != "link":
                segments.append(gap)

        wave = np.concatenate(segments) if segments else np.zeros(sr)
        # Normalize to -18 dBFS
        peak = np.max(np.abs(wave))
        if peak > 0:
            wave = wave / peak * 0.25
        return wave.astype(np.float32)

    def save(self, tokens: list[str], path: Path) -> Path:
        wave = self.synthesize(tokens)
        sf.write(str(path), wave, self.p.sample_rate)
        return path


# ── ResponsePlanner ───────────────────────────────────────────────────────────

def _build_response_table() -> dict[str, dict]:
    from whale_engine import (
        GREETING_CALL, ALARM_CALL, HUMPBACK_SONG_CANON, HUMPBACK_SONG_EXTENDED,
        SPERM_WHALE_CODA_A, SPERM_WHALE_CODA_I, SPERM_WHALE_CODA_I_LONG,
        SPERM_WHALE_DIPHTHONG, SPERM_WHALE_CLAN_EXCHANGE,
        ORCA_PULSED_CALL, ORCA_POD_EXCHANGE, ORCA_SOCIAL_BONDING,
        ORCA_ALARM_CALL, ORCA_HUNT_SEQUENCE, ORCA_CROSS_POD,
        ORCA_SIGNATURE_WHISTLE, ORCA_CLICK_TRAIN,
    )
    return {
        "greeting_call": {
            "response": "greeting_call",
            Species.HUMPBACK:    GREETING_CALL,
            Species.SPERM_WHALE: SPERM_WHALE_CODA_I,
            Species.ORCA:        ORCA_PULSED_CALL,
        },
        "question": {
            "response": "greeting_call",
            Species.HUMPBACK:    GREETING_CALL,
            Species.SPERM_WHALE: SPERM_WHALE_CODA_A,
            Species.ORCA:        ORCA_PULSED_CALL,
        },
        "alarm_call": {
            "response": "alarm_call",
            Species.HUMPBACK:    ALARM_CALL,
            Species.SPERM_WHALE: ["init", "evalf", "paradox", "fix"],
            Species.ORCA:        ORCA_ALARM_CALL,
        },
        "narrative": {
            "response": "song",
            Species.HUMPBACK:    HUMPBACK_SONG_CANON,
            Species.SPERM_WHALE: SPERM_WHALE_CLAN_EXCHANGE,
            Species.ORCA:        ORCA_SOCIAL_BONDING,
        },
        "song": {
            "response": "song",
            Species.HUMPBACK:    HUMPBACK_SONG_CANON,
            Species.SPERM_WHALE: SPERM_WHALE_CLAN_EXCHANGE,
            Species.ORCA:        ORCA_POD_EXCHANGE,
        },
        "dialetheic_expression": {
            "response": "dialetheic_expression",
            Species.HUMPBACK:    ["init", "paradox", "rep", "paradox", "fix", "anc"],
            Species.SPERM_WHALE: ["init", "paradox", "rep", "paradox", "fix", "anc"],
            Species.ORCA:        ORCA_CROSS_POD,
        },
        "coda_exchange": {
            "response": "coda_exchange",
            Species.HUMPBACK:    HUMPBACK_SONG_EXTENDED,
            Species.SPERM_WHALE: SPERM_WHALE_CLAN_EXCHANGE,
            Species.ORCA:        ORCA_POD_EXCHANGE,
        },
        "phonological_contrast": {
            "response": "phonological_contrast",
            Species.HUMPBACK:    ["init", "up", "split", "dn", "fuse", "link", "fix", "anc"],
            Species.SPERM_WHALE: SPERM_WHALE_DIPHTHONG,
            Species.ORCA:        ORCA_SIGNATURE_WHISTLE,
        },
        "sustained_social_call": {
            "response": "sustained_social_call",
            Species.HUMPBACK:    ["init", "up", "evalt", "rep", "link", "fix", "anc"],
            Species.SPERM_WHALE: SPERM_WHALE_CODA_I_LONG,
            Species.ORCA:        ORCA_SOCIAL_BONDING,
        },
        "pod_dialect": {
            "response": "pod_dialect",
            Species.HUMPBACK:    HUMPBACK_SONG_CANON,
            Species.SPERM_WHALE: SPERM_WHALE_CODA_A,
            Species.ORCA:        ORCA_POD_EXCHANGE,
        },
        "hunting_call": {
            "response": "coordination_signal",
            Species.HUMPBACK:    ALARM_CALL,
            Species.SPERM_WHALE: ["init", "evalf", "rep", "paradox", "fix", "anc"],
            Species.ORCA:        ORCA_HUNT_SEQUENCE,
        },
        "coordination_signal": {
            "response": "coordination_signal",
            Species.HUMPBACK:    ALARM_CALL,
            Species.SPERM_WHALE: SPERM_WHALE_CODA_I,
            Species.ORCA:        ORCA_HUNT_SEQUENCE,
        },
        "social_bonding_call": {
            "response": "social_bonding_call",
            Species.HUMPBACK:    HUMPBACK_SONG_CANON,
            Species.SPERM_WHALE: SPERM_WHALE_CODA_I_LONG,
            Species.ORCA:        ORCA_SOCIAL_BONDING,
        },
        "cross_pod_dialect": {
            "response": "cross_pod_dialect",
            Species.HUMPBACK:    HUMPBACK_SONG_EXTENDED,
            Species.SPERM_WHALE: SPERM_WHALE_CLAN_EXCHANGE,
            Species.ORCA:        ORCA_CROSS_POD,
        },
        "echolocation_probe": {
            "response": "greeting_call",
            Species.HUMPBACK:    GREETING_CALL,
            Species.SPERM_WHALE: SPERM_WHALE_CODA_A,
            Species.ORCA:        ORCA_CLICK_TRAIN,
        },
    }


class ResponsePlanner:
    """Maps an incoming expression to an appropriate response token sequence.

    Strategy: preserve structural register (greeting→greeting, alarm→alarm),
    escalate complexity for narratives/songs, mirror phonological contrasts.
    Falls back to greeting_call if no specific strategy is found.
    """

    def __init__(self) -> None:
        self._table = _build_response_table()

    def plan(self, expression: str, species: Species) -> tuple[str, list[str]]:
        """Return (response_expression_name, token_list) for the given input."""
        entry = self._table.get(expression, self._table["greeting_call"])
        response_name = entry["response"]
        tokens = entry.get(species, entry[Species.HUMPBACK])
        return response_name, tokens

    @property
    def known_expressions(self) -> list[str]:
        return sorted(self._table.keys())


# ── WhaleSpeaker ──────────────────────────────────────────────────────────────

class WhaleSpeaker:
    """Unified entry point for cetacean response synthesis.

    speak(expression, species, output_path)
        Synthesize a specific expression as whale audio.

    respond(wav_path, species, output_path)
        Analyze incoming WAV, plan structurally-appropriate response, synthesize.
    """

    def __init__(self) -> None:
        self._planner = ResponsePlanner()

    def speak(self, expression: str, species: Species,
              output_path: Optional[Path] = None) -> dict:
        """Synthesize expression as species-appropriate audio.

        Args:
            expression:  An archetype name (e.g. "song", "greeting_call",
                         "coda_exchange") or a space-separated token sequence
                         (e.g. "init up rep fix anc").
            species:     Target species.
            output_path: Where to write the WAV. Defaults to
                         ./<species>_<expression>.wav in cwd.

        Returns:
            Dict with tokens, output_path, duration_s.
        """
        synth = CetaceanSynthesizer(species)

        # Determine token list: archetype name or raw token sequence
        _, tokens = self._planner.plan(expression, species)
        if output_path is None:
            output_path = Path(f"{species.value}_{expression.replace(' ', '_')}.wav")

        synth.save(tokens, output_path)
        wave = synth.synthesize(tokens)
        duration_s = len(wave) / synth.p.sample_rate

        print(f"  species:    {species.value}")
        print(f"  expression: {expression}")
        print(f"  tokens:     {' → '.join(tokens)}")
        print(f"  duration:   {duration_s:.2f}s  ({len(tokens)} units)")
        print(f"  output:     {output_path}")

        return {
            "tokens": tokens,
            "output_path": output_path,
            "duration_s": duration_s,
            "species": species.value,
            "expression": expression,
        }

    def respond(self, wav_path: Path | str, species: Species,
                output_path: Optional[Path] = None,
                params=None, verbose: bool = True) -> dict:
        """Analyze incoming WAV and synthesize a structural response.

        Full pipeline:
          WAV → whale_audio → structural signature → expression match
          → ResponsePlanner → token sequence → CetaceanSynthesizer → WAV

        Args:
            wav_path:    Path to incoming whale vocalization WAV.
            species:     Target species for the response.
            output_path: Where to write response WAV.
            params:      ClassifierParams for whale_audio (None = defaults).
            verbose:     Print analysis output.

        Returns:
            Dict with analysis, incoming expression, response tokens, output path.
        """
        from whale_audio import translate_wav

        wav_path = Path(wav_path)
        if output_path is None:
            output_path = wav_path.parent / f"{wav_path.stem}_response_{species.value}.wav"

        # Step 1: Analyze incoming with species-appropriate classifier
        if verbose:
            print(f"\n── Incoming: {wav_path.name} ──")
        result = translate_wav(wav_path, params=params, verbose=verbose,
                               species=species.value)
        incoming_expression = result["ranked"][0][0]
        incoming_distance = result["ranked"][0][1]

        # Step 2: Plan response
        response_name, response_tokens = self._planner.plan(incoming_expression, species)

        if verbose:
            print(f"\n── Response plan ──")
            print(f"   incoming:  {incoming_expression}  (d={incoming_distance:.4f})")
            print(f"   response:  {response_name}")
            print(f"   tokens:    {' → '.join(response_tokens)}")
            print(f"   species:   {species.value}")

        # Step 3: Synthesize
        synth = CetaceanSynthesizer(species)
        synth.save(response_tokens, output_path)
        wave = synth.synthesize(response_tokens)
        duration_s = len(wave) / synth.p.sample_rate

        if verbose:
            print(f"\n── Synthesized response ──")
            print(f"   duration:  {duration_s:.2f}s")
            print(f"   output:    {output_path}")

        return {
            "incoming_wav": str(wav_path),
            "incoming_expression": incoming_expression,
            "incoming_distance": incoming_distance,
            "response_expression": response_name,
            "response_tokens": response_tokens,
            "response_wav": str(output_path),
            "duration_s": duration_s,
            "species": species.value,
            "analysis": result,
        }

    @property
    def expressions(self) -> list[str]:
        return self._planner.known_expressions


# ── CLI ───────────────────────────────────────────────────────────────────────

def _list_expressions() -> None:
    speaker = WhaleSpeaker()
    print("Available expression archetypes:")
    for name in speaker.expressions:
        print(f"  {name}")
    print("\nAvailable species:")
    for s in Species:
        print(f"  {s.value}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Cetacean Response Synthesizer — whale_speaker.py"
    )
    parser.add_argument(
        "--species", "-s",
        choices=[s.value for s in Species],
        default="humpback",
        help="Target species for synthesis (default: humpback)",
    )
    parser.add_argument(
        "--expression", "-e",
        metavar="EXPR",
        help="Expression archetype to synthesize (manual mode)",
    )
    parser.add_argument(
        "--respond", "-r",
        metavar="WAV",
        help="Incoming WAV file to analyze and respond to (response mode)",
    )
    parser.add_argument(
        "--output", "-o",
        metavar="PATH",
        help="Output WAV path (default: auto-named in current directory)",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available expressions and species",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress analysis output",
    )

    args = parser.parse_args()

    if args.list:
        _list_expressions()
        return

    if not args.expression and not args.respond:
        parser.print_help()
        sys.exit(1)

    species = Species(args.species)
    output = Path(args.output) if args.output else None
    speaker = WhaleSpeaker()

    if args.expression:
        print(f"── Speaking: {args.expression} ({species.value}) ──")
        speaker.speak(args.expression, species, output_path=output)
    elif args.respond:
        speaker.respond(
            Path(args.respond), species,
            output_path=output,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()
