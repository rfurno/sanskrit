"""
Phoneme Aligner & Sanskrit Phonetic Utilities.

Provides:
- Robust Devanagari → IAST romanization (via indic-transliteration)
- Custom Sanskrit phoneme inventory and simple grapheme-to-phoneme rules
- Loading of mantra phoneme sequences with timing hints
- DTW-based time alignment between reference and user audio features
- Vowel length (mātrā) detection and scoring helpers
- Approximate forced alignment of phonemes to audio frames
"""

from __future__ import annotations
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from rapidfuzz import fuzz

from config import (
    MANTRAS_DIR,
    SANSKRIT_VOWELS,
    SANSKRIT_CONSONANTS,
    GAYATRI_PHONEME_SEQUENCE,
    METRIC_CONFIG,
)
from core.audio_processor import (
    load_audio,
    extract_mfcc,
    extract_rms_energy,
    compute_dtw_distance,
    trim_silence,
    extract_prosodic_features,
)

# Optional but excellent for Sanskrit
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    INDIC_TRANSLITERATION_AVAILABLE = True
except ImportError:
    INDIC_TRANSLITERATION_AVAILABLE = False


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Phoneme:
    symbol: str
    is_vowel: bool
    is_long: bool
    expected_relative_duration: float
    notes: str = ""

    def __repr__(self) -> str:
        length = "long" if self.is_long else ("short" if self.is_vowel else "")
        return f"Phoneme({self.symbol!r}, {length})"


@dataclass
class MantraData:
    id: str
    name: str
    devanagari: str
    iast: str
    translation: str
    phoneme_sequence: List[Phoneme]
    reference_audio_path: Path
    approx_duration: float
    guidance: List[str]
    common_mistakes: List[str]
    raw_json: Dict[str, Any]


@dataclass
class AlignmentResult:
    """Result of aligning user audio to reference."""
    ref_mfcc: np.ndarray
    user_mfcc: np.ndarray
    dtw_distance: float
    alignment_path: np.ndarray  # (N, 2) indices into ref and user
    ref_duration: float
    user_duration: float
    ref_features: Dict[str, Any]
    user_features: Dict[str, Any]


# =============================================================================
# ROMANIZATION & DEVANAGARI HELPERS
# =============================================================================

def to_iast(text: str) -> str:
    """Convert Devanagari (or other Indic) to IAST romanization."""
    if INDIC_TRANSLITERATION_AVAILABLE:
        return transliterate(text, sanscript.DEVANAGARI, sanscript.IAST)
    # Fallback: very rough manual mapping for common Gayatri chars (expand as needed)
    replacements = {
        "ॐ": "oṃ", "।": " | ", "॥": " || ",
        "अ": "a", "आ": "ā", "इ": "i", "ई": "ī", "उ": "u", "ऊ": "ū",
        "ऋ": "ṛ", "ॠ": "ṝ", "ऌ": "ḷ",
        "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
        "ं": "ṃ", "ः": "ḥ", "ँ": "m̐",
        "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ṅ",
        "च": "c", "छ": "ch", "ज": "j", "झ": "jh", "ञ": "ñ",
        "ट": "ṭ", "ठ": "ṭh", "ड": "ḍ", "ढ": "ḍh", "ण": "ṇ",
        "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
        "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
        "य": "y", "र": "r", "ल": "l", "व": "v",
        "श": "ś", "ष": "ṣ", "स": "s", "ह": "h",
    }
    out = text
    for dev, rom in replacements.items():
        out = out.replace(dev, rom)
    return out


def normalize_for_comparison(text: str) -> str:
    """Lowercase, remove punctuation, collapse spaces for fuzzy matching."""
    text = text.lower()
    text = re.sub(r"[।॥|.,;:!?।\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def iast_to_phonemes(iast: str) -> List[str]:
    """
    Very lightweight IAST → phoneme tokenization.
    Splits into approximate Sanskrit phonemes/syllables.
    Good enough for alignment hints and feedback on Gayatri-scale mantras.
    """
    iast = iast.lower().strip()
    # Common conjuncts and special cases first
    special = ["oṃ", "aḥ", "iḥ", "uḥ", "eḥ", "oḥ", "āḥ", "īḥ", "ūḥ",
               "kh", "gh", "ch", "jh", "ṭh", "ḍh", "th", "dh", "ph", "bh",
               "ś", "ṣ", "ṇ", "ñ", "ṅ", "ṛ", "ṝ", "ḷ", "ṃ", "ḥ"]

    phonemes: List[str] = []
    i = 0
    while i < len(iast):
        matched = False
        for sp in special:
            if iast[i : i + len(sp)] == sp:
                phonemes.append(sp)
                i += len(sp)
                matched = True
                break
        if matched:
            continue

        char = iast[i]
        if char in "aeiouāīūṛṝḷeaiou":
            phonemes.append(char)
        elif char.isalpha():
            # Single consonant or other
            phonemes.append(char)
        # ignore spaces/punct
        i += 1

    return phonemes


# =============================================================================
# MANTRA LOADING
# =============================================================================

def load_mantra(mantra_id: str) -> MantraData:
    """Load a mantra definition from JSON + enrich with Phoneme objects."""
    path = MANTRAS_DIR / f"{mantra_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Mantra definition not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Build Phoneme list (prefer explicit sequence in JSON)
    phoneme_list: List[Phoneme] = []
    seq = data.get("phoneme_sequence", GAYATRI_PHONEME_SEQUENCE)

    for entry in seq:
        if isinstance(entry, (list, tuple)):
            symbol = str(entry[0])
            rel_dur = float(entry[1]) if len(entry) > 1 else 1.0
            is_vowel = bool(entry[2]) if len(entry) > 2 else (symbol[0] in "aeiouāīūṛṝe")
            is_long = bool(entry[3]) if len(entry) > 3 else (symbol in SANSKRIT_VOWELS["long"] or symbol in {"oṃ", "āṃ"})
            notes = str(entry[4]) if len(entry) > 4 else ""
        else:
            symbol = str(entry)
            rel_dur = 1.0
            is_vowel = symbol[0] in "aeiouāīūṛṝḷe"
            is_long = symbol in SANSKRIT_VOWELS["long"] or symbol in {"oṃ"}
            notes = ""

        phoneme_list.append(Phoneme(symbol, is_vowel, is_long, rel_dur, notes))

    ref_audio = Path(data.get("reference_audio", f"audio/references/{mantra_id}.wav"))
    if not ref_audio.is_absolute():
        ref_audio = (MANTRAS_DIR.parent / ref_audio).resolve()

    return MantraData(
        id=data["id"],
        name=data["name"],
        devanagari=data["devanagari"],
        iast=data["iast"],
        translation=data.get("english_translation", ""),
        phoneme_sequence=phoneme_list,
        reference_audio_path=ref_audio,
        approx_duration=float(data.get("approximate_duration_seconds", 36.0)),
        guidance=data.get("sanskrit_specific_guidance", []),
        common_mistakes=data.get("common_mistakes", []),
        raw_json=data,
    )


def list_available_mantras() -> List[str]:
    """Return list of mantra IDs available in mantras/ directory."""
    return sorted(p.stem for p in MANTRAS_DIR.glob("*.json"))


# =============================================================================
# ALIGNMENT + VOWEL LENGTH ANALYSIS
# =============================================================================

def align_reference_and_user(
    ref_audio_path: Path,
    user_audio: np.ndarray,
    user_sr: int,
) -> AlignmentResult:
    """
    Core alignment routine.
    1. Load + preprocess reference
    2. Extract MFCC + prosodic features for both
    3. Compute DTW on MFCCs
    """
    ref_y, ref_sr = load_audio(ref_audio_path, target_sr=user_sr)

    # Preprocess both
    ref_y = trim_silence_if_needed(ref_y)
    user_y = trim_silence_if_needed(user_audio)

    ref_mfcc = extract_mfcc(ref_y, ref_sr)
    user_mfcc = extract_mfcc(user_y, user_sr)

    dtw_dist, path = compute_dtw_distance(ref_mfcc, user_mfcc)

    ref_features = extract_prosodic_features(ref_y, ref_sr)
    user_features = extract_prosodic_features(user_y, user_sr)

    return AlignmentResult(
        ref_mfcc=ref_mfcc,
        user_mfcc=user_mfcc,
        dtw_distance=dtw_dist,
        alignment_path=path,
        ref_duration=ref_features["duration"],
        user_duration=user_features["duration"],
        ref_features=ref_features,
        user_features=user_features,
    )


def trim_silence_if_needed(y: np.ndarray) -> np.ndarray:
    """Light wrapper around audio_processor trim."""
    return trim_silence(y)


def estimate_vowel_segments(
    rms: np.ndarray,
    pitch_contour: Dict[str, np.ndarray],
    sr: int,
    hop: int,
) -> List[Dict[str, Any]]:
    """
    Heuristic detection of sustained vowel regions.
    Returns list of {"start_frame", "end_frame", "duration_s", "mean_pitch", "is_long_candidate"}
    """
    hop_s = hop / sr
    energy_thresh = np.percentile(rms, 25) * 0.6
    voiced = pitch_contour["voiced"]

    segments = []
    in_vowel = False
    start = 0

    for i in range(len(rms)):
        is_voiced = bool(voiced[i]) if i < len(voiced) else False
        high_energy = rms[i] > energy_thresh

        if not in_vowel and is_voiced and high_energy:
            in_vowel = True
            start = i
        elif in_vowel and (not is_voiced or not high_energy):
            dur_s = (i - start) * hop_s
            if dur_s > 0.08:  # ignore micro segments
                mean_p = float(np.nanmean(pitch_contour["pitch"][start:i])) if start < i else 0.0
                segments.append({
                    "start_frame": start,
                    "end_frame": i,
                    "duration_s": dur_s,
                    "mean_pitch": mean_p,
                    "is_long_candidate": dur_s > (METRIC_CONFIG.long_vowel_min_duration_ms / 1000.0),
                })
            in_vowel = False

    return segments


def score_vowel_length_accuracy(
    ref_alignment: AlignmentResult,
    user_segments: List[Dict[str, Any]],
    expected_phonemes: List[Phoneme],
) -> Dict[str, Any]:
    """
    Compare detected vowel durations in user against expected long/short from mantra.
    Returns score (0-100) + detailed mismatches.
    """
    ref_dur = ref_alignment.ref_duration
    user_dur = ref_alignment.user_duration
    tempo_ratio = user_dur / max(ref_dur, 0.1)

    long_expected = [p for p in expected_phonemes if p.is_vowel and p.is_long]
    short_expected = [p for p in expected_phonemes if p.is_vowel and not p.is_long]

    # Very approximate: assume user long vowels should be ~1.7-2.2x short vowels after tempo norm
    user_long_durs = [s["duration_s"] for s in user_segments if s["is_long_candidate"]]
    user_short_durs = [s["duration_s"] for s in user_segments if not s["is_long_candidate"]]

    if not user_long_durs or not user_short_durs:
        # Fallback: use overall energy envelope variation as proxy
        base_score = 62.0
        return {
            "score": base_score,
            "comment": "Could not reliably segment vowels. Check recording quality and volume.",
            "long_vowels_detected": len(user_long_durs),
            "short_vowels_detected": len(user_short_durs),
            "tempo_ratio": round(tempo_ratio, 3),
        }

    avg_long = float(np.mean(user_long_durs))
    avg_short = float(np.mean(user_short_durs))
    observed_ratio = avg_long / max(avg_short, 0.01)

    ideal_ratio = 1.95
    ratio_error = abs(observed_ratio - ideal_ratio) / ideal_ratio

    # Penalize if tempo is wildly off (user spoke much faster/slower)
    tempo_penalty = min(18.0, abs(tempo_ratio - 1.0) * 35)

    vowel_score = max(35.0, 100.0 - (ratio_error * 55) - tempo_penalty)

    comment = (
        f"Observed long/short vowel ratio ≈ {observed_ratio:.2f} (ideal ~1.9-2.1). "
        f"Tempo ratio vs reference: {tempo_ratio:.2f}×."
    )
    if observed_ratio < 1.45:
        comment += " Long vowels are too short — hold mātrā fully."
    elif observed_ratio > 2.6:
        comment += " Long vowels may be overly stretched."

    return {
        "score": round(float(vowel_score), 1),
        "comment": comment,
        "long_vowels_detected": len(user_long_durs),
        "short_vowels_detected": len(user_short_durs),
        "observed_ratio": round(observed_ratio, 2),
        "tempo_ratio": round(tempo_ratio, 3),
    }


def compute_text_similarity(
    expected_iast: str,
    asr_transcript: Optional[str],
) -> Dict[str, Any]:
    """Fuzzy + normalized similarity between expected IAST and ASR output."""
    if not asr_transcript:
        return {"score": 55.0, "wer": None, "comment": "No ASR transcript available."}

    norm_exp = normalize_for_comparison(expected_iast)
    norm_asr = normalize_for_comparison(asr_transcript)

    # Rapidfuzz ratio (0-100)
    ratio = fuzz.ratio(norm_exp, norm_asr)
    token_sort = fuzz.token_sort_ratio(norm_exp, norm_asr)

    combined = (ratio * 0.55 + token_sort * 0.45)

    # Convert to 0-100 pronunciation-friendly score (penalize very low matches more)
    text_score = max(30.0, min(98.0, combined * 0.92))

    return {
        "score": round(text_score, 1),
        "fuzz_ratio": int(ratio),
        "token_sort": int(token_sort),
        "comment": f"Text similarity: {text_score:.0f}/100 (ASR may struggle with Sanskrit).",
    }


def estimate_phoneme_alignment_quality(
    alignment: AlignmentResult,
    expected_phonemes: List[Phoneme],
) -> Dict[str, Any]:
    """
    Use DTW path + energy/pitch variation along path to estimate how well
    local acoustic events match expected phoneme sequence.
    This is a proxy for "pronunciation accuracy" at phoneme level.
    """
    path = alignment.alignment_path
    ref_rms = alignment.ref_features["rms_envelope"]
    user_rms = alignment.user_features["rms_envelope"]

    # Sample energy at corresponding frames along path
    ref_energy_along = []
    user_energy_along = []
    for r_idx, u_idx in path:
        if r_idx < len(ref_rms) and u_idx < len(user_rms):
            ref_energy_along.append(ref_rms[r_idx])
            user_energy_along.append(user_rms[u_idx])

    if not ref_energy_along:
        return {"score": 60.0, "comment": "Insufficient alignment data."}

    ref_e = np.array(ref_energy_along)
    user_e = np.array(user_energy_along)

    # Correlation of energy envelopes after alignment (good proxy for timing of consonants/vowels)
    if len(ref_e) > 3:
        corr = float(np.corrcoef(ref_e, user_e)[0, 1])
    else:
        corr = 0.5

    # Penalize large DTW distance
    dtw = alignment.dtw_distance
    dtw_penalty = min(32.0, (dtw - 0.35) * 55) if dtw > 0.35 else 0.0

    base = 55.0 + (corr * 32.0)
    phoneme_score = max(38.0, min(96.0, base - dtw_penalty))

    comment = (
        f"Acoustic pattern match (DTW={dtw:.3f}, energy_corr={corr:.2f}). "
    )
    if dtw > 0.72:
        comment += "Significant timing or articulation differences detected."
    elif dtw < 0.48:
        comment += "Very good local acoustic alignment."

    return {
        "score": round(phoneme_score, 1),
        "dtw_distance": round(dtw, 3),
        "energy_correlation": round(corr, 3),
        "comment": comment,
    }
