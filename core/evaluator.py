"""
Evaluator - The heart of SanskritMantraPronunciationCoach.

Computes all eight configurable evaluation dimensions with:
- Normalized 0-100 scores
- Weighted contribution to overall
- Specific, actionable feedback comments
- Detailed intermediate measurements for visualizations and reports

All weights and thresholds live in config.py so the system is fully tunable.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import time

import numpy as np

from config import (
    get_normalized_weights,
    THRESHOLDS,
    METRIC_CONFIG,
    ASR_CONFIG,
    EVALUATION_WEIGHTS,
)
from core.audio_processor import (
    load_audio,
    extract_rms_energy,
    extract_pitch_parselmouth,
    extract_prosodic_features,
    play_audio,
)
from core.phoneme_aligner import (
    load_mantra,
    MantraData,
    AlignmentResult,
    align_reference_and_user,
    estimate_vowel_segments,
    score_vowel_length_accuracy,
    compute_text_similarity,
    estimate_phoneme_alignment_quality,
)


@dataclass
class AttributeScore:
    name: str
    score: float
    weighted_contribution: float
    comment: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationResult:
    mantra_id: str
    mantra_name: str
    timestamp: str
    user_audio_path: Optional[str]
    reference_audio_path: str

    overall_score: float
    attribute_scores: Dict[str, AttributeScore]

    summary: str
    strengths: List[str]
    weaknesses: List[str]
    sanskrit_tips: List[str]

    alignment: Optional[Dict[str, Any]] = None  # lightweight serializable view
    raw_measurements: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert nested AttributeScore
        d["attribute_scores"] = {k: v.to_dict() for k, v in self.attribute_scores.items()}
        return d

    def save_report(self, path: Optional[Path] = None) -> Path:
        """Save JSON report for later review or ML dataset creation."""
        from config import REPORTS_DIR
        if path is None:
            ts = self.timestamp.replace(":", "-").replace(" ", "_")
            path = REPORTS_DIR / f"{self.mantra_id}_{ts}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return path


def _label(score: float) -> str:
    return THRESHOLDS.get_label(score)


def _weighted(score: float, weight: float) -> float:
    return round(score * weight, 2)


# =============================================================================
# INDIVIDUAL METRIC COMPUTATIONS
# =============================================================================

def _score_pronunciation_accuracy(
    alignment: AlignmentResult,
    mantra: MantraData,
    asr_transcript: Optional[str] = None,
) -> AttributeScore:
    """
    Pronunciation Accuracy (phoneme-level fidelity).
    Combines acoustic DTW match + optional text similarity from ASR.
    Sanskrit focus: retroflexes, aspiration, conjuncts, visarga.
    """
    phoneme_quality = estimate_phoneme_alignment_quality(alignment, mantra.phoneme_sequence)

    text_sim = compute_text_similarity(mantra.iast, asr_transcript) if ASR_CONFIG.enabled else None

    # Base acoustic score
    acoustic = phoneme_quality["score"]

    if text_sim and text_sim.get("score"):
        # Blend (text is noisy for Sanskrit)
        blend = (acoustic * (1 - METRIC_CONFIG.asr_weight_in_pronunciation) +
                 text_sim["score"] * METRIC_CONFIG.asr_weight_in_pronunciation)
        score = round(blend, 1)
        comment = f"{phoneme_quality['comment']} ASR similarity contributed modestly."
    else:
        score = acoustic
        comment = phoneme_quality["comment"]

    weight = get_normalized_weights()["pronunciation_accuracy"]
    return AttributeScore(
        name="Pronunciation Accuracy",
        score=score,
        weighted_contribution=_weighted(score, weight),
        comment=comment,
        details={
            "dtw_distance": phoneme_quality.get("dtw_distance"),
            "energy_correlation": phoneme_quality.get("energy_correlation"),
            "text_similarity": text_sim,
        },
    )


def _score_vowel_length(
    alignment: AlignmentResult,
    mantra: MantraData,
) -> AttributeScore:
    """Vowel length (mātrā) accuracy - one of the most critical Sanskrit dimensions."""
    user_rms = alignment.user_features["rms_envelope"]
    user_pitch = alignment.user_features["pitch_contour"]
    hop = 512  # matches AUDIO_CONFIG.hop_length

    # Reconstruct approximate sample rate from duration and frame count
    approx_sr = int(len(user_rms) * hop / max(alignment.user_duration, 0.1))

    user_segments = estimate_vowel_segments(
        user_rms, user_pitch, approx_sr, hop
    )

    result = score_vowel_length_accuracy(alignment, user_segments, mantra.phoneme_sequence)

    weight = get_normalized_weights()["vowel_length_accuracy"]
    return AttributeScore(
        name="Vowel Length Accuracy",
        score=result["score"],
        weighted_contribution=_weighted(result["score"], weight),
        comment=result["comment"],
        details=result,
    )


def _score_pitch_stability(
    alignment: AlignmentResult,
) -> AttributeScore:
    """Tone / Pitch stability. Lower variance on sustained vowels is better for traditional chanting."""
    ref_p = alignment.ref_features["pitch_contour"]
    user_p = alignment.user_features["pitch_contour"]

    ref_std = ref_p["std_pitch"] or 1.0
    user_std = user_p["std_pitch"] or 1.0

    # Compare stability (lower std relative to mean is better). Also penalize large octave jumps.
    ref_cv = ref_std / max(ref_p["mean_pitch"], 1)
    user_cv = user_std / max(user_p["mean_pitch"], 1)

    stability_ratio = user_cv / max(ref_cv, 0.001)

    # Base score from how much more variable the user is
    if stability_ratio <= 1.15:
        base = 92.0
    elif stability_ratio <= 1.6:
        base = 82.0 - (stability_ratio - 1.15) * 28
    else:
        base = max(48.0, 82.0 - (stability_ratio - 1.15) * 22)

    # Jitter penalty using raw std (in Hz converted roughly to cents is better but we approximate)
    jitter_penalty = min(22.0, (user_std - ref_std) * 0.9)
    score = max(42.0, base - jitter_penalty)

    comment = (
        f"Pitch variation (user CV={user_cv:.3f} vs ref {ref_cv:.3f}). "
    )
    if user_std > ref_std * 1.8:
        comment += "Too much pitch wobble — aim for steadier sustained vowels."
    elif user_std < ref_std * 0.6:
        comment += "Very stable — excellent if intentional, but traditional chanting has subtle life."

    weight = get_normalized_weights()["tone_pitch_stability"]
    return AttributeScore(
        name="Tone / Pitch Stability",
        score=round(score, 1),
        weighted_contribution=_weighted(score, weight),
        comment=comment,
        details={
            "user_pitch_std_hz": round(user_std, 1),
            "ref_pitch_std_hz": round(ref_std, 1),
            "user_mean_pitch": round(user_p["mean_pitch"], 1),
        },
    )


def _score_tempo(
    alignment: AlignmentResult,
) -> AttributeScore:
    """Speed / Tempo match (overall duration + rhythm feel)."""
    ref_d = alignment.ref_duration
    user_d = alignment.user_duration
    ratio = user_d / max(ref_d, 0.1)

    tolerance = METRIC_CONFIG.tempo_tolerance_percent / 100.0
    deviation = abs(ratio - 1.0)

    if deviation <= tolerance * 0.6:
        score = 93.0
    elif deviation <= tolerance:
        score = 82.0 - (deviation - tolerance * 0.6) * 70
    else:
        score = max(45.0, 82.0 - (deviation - tolerance) * 95)

    comment = f"Duration ratio: {ratio:.2f}× reference ({ref_d:.1f}s → {user_d:.1f}s). "
    if ratio > 1.22:
        comment += "You are chanting noticeably slower. Good for clarity, but watch rhythm."
    elif ratio < 0.82:
        comment += "Quite fast — ensure all mātrā and visarga are fully articulated."

    weight = get_normalized_weights()["speed_tempo"]
    return AttributeScore(
        name="Speed / Tempo",
        score=round(score, 1),
        weighted_contribution=_weighted(score, weight),
        comment=comment,
        details={"duration_ratio": round(ratio, 3), "ref_duration": round(ref_d, 2)},
    )


def _score_smoothness_fluency(
    alignment: AlignmentResult,
) -> AttributeScore:
    """Smoothness / Fluency: pauses, hesitations, flow between phrases."""
    user_rms = alignment.user_features["rms_envelope"]
    user_dur = alignment.user_duration

    # Detect unnatural low-energy gaps
    hop_s = user_dur / max(len(user_rms), 1)
    energy_thresh = np.percentile(user_rms, 18)

    pauses = []
    in_pause = False
    start = 0
    for i, e in enumerate(user_rms):
        if e < energy_thresh and not in_pause:
            in_pause = True
            start = i
        elif e >= energy_thresh and in_pause:
            dur = (i - start) * hop_s
            if dur > (METRIC_CONFIG.breath_min_silence_ms / 1000.0):
                pauses.append(dur)
            in_pause = False

    num_pauses = len(pauses)
    long_pauses = [p for p in pauses if p > METRIC_CONFIG.max_acceptable_pause_ms / 1000.0]

    penalty = len(long_pauses) * METRIC_CONFIG.unnatural_pause_penalty_per
    base = 88.0 - penalty
    score = max(48.0, min(95.0, base))

    comment = f"Detected ~{num_pauses} significant pauses. "
    if long_pauses:
        comment += f"{len(long_pauses)} were longer than ideal ({METRIC_CONFIG.max_acceptable_pause_ms}ms). Maintain flow between phrases."
    else:
        comment += "Good phrase connectivity."

    weight = get_normalized_weights()["smoothness_fluency"]
    return AttributeScore(
        name="Smoothness / Fluency",
        score=round(score, 1),
        weighted_contribution=_weighted(score, weight),
        comment=comment,
        details={"num_pauses": num_pauses, "long_pauses": len(long_pauses)},
    )


def _score_volume_consistency(
    alignment: AlignmentResult,
) -> AttributeScore:
    """Volume Consistency."""
    user_rms = alignment.user_features["rms_envelope"]
    if len(user_rms) < 3:
        score = 70.0
        comment = "Very short recording."
    else:
        # Use dB-like variation
        rms_db = 20 * np.log10(np.maximum(user_rms, 1e-6))
        std_db = float(np.std(rms_db))
        dyn_range = float(np.max(rms_db) - np.min(rms_db))

        if std_db < METRIC_CONFIG.volume_variation_max_db * 0.55:
            score = 91.0
        elif std_db < METRIC_CONFIG.volume_variation_max_db:
            score = 82.0 - (std_db - METRIC_CONFIG.volume_variation_max_db * 0.55) * 1.8
        else:
            score = max(52.0, 82.0 - (std_db - METRIC_CONFIG.volume_variation_max_db) * 2.8)

        comment = f"RMS variation ≈ {std_db:.1f} dB (target < {METRIC_CONFIG.volume_variation_max_db} dB). "

        if std_db > METRIC_CONFIG.volume_variation_max_db:
            comment += "Try to keep amplitude more even across the whole mantra."
        else:
            comment += "Good dynamic control."

    weight = get_normalized_weights()["volume_consistency"]
    return AttributeScore(
        name="Volume Consistency",
        score=round(score, 1),
        weighted_contribution=_weighted(score, weight),
        comment=comment,
        details={"rms_std_db": round(std_db, 2) if 'std_db' in locals() else None},
    )


def _score_breath_control(
    alignment: AlignmentResult,
) -> AttributeScore:
    """Breath Control: placement and length of breathing gaps."""
    user_rms = alignment.user_features["rms_envelope"]
    user_dur = alignment.user_duration
    hop_s = user_dur / max(len(user_rms), 1)

    energy_thresh = np.percentile(user_rms, 12)
    min_silence_frames = int(METRIC_CONFIG.breath_min_silence_ms / 1000 / hop_s)

    breath_durs = []
    i = 0
    while i < len(user_rms):
        if user_rms[i] < energy_thresh:
            j = i
            while j < len(user_rms) and user_rms[j] < energy_thresh:
                j += 1
            dur = (j - i) * hop_s
            if dur >= (METRIC_CONFIG.breath_min_silence_ms / 1000.0):
                breath_durs.append(dur)
            i = j
        else:
            i += 1

    bad_breaths = [d for d in breath_durs if d > METRIC_CONFIG.breath_max_natural_ms / 1000.0]
    score = 88.0 - len(bad_breaths) * 11.0
    score = max(50.0, min(94.0, score))

    comment = f"Detected {len(breath_durs)} breathing pauses. "
    if bad_breaths:
        comment += f"{len(bad_breaths)} were excessively long or poorly placed."
    else:
        comment += "Breath placement looks natural."

    weight = get_normalized_weights()["breath_control"]
    return AttributeScore(
        name="Breath Control",
        score=round(score, 1),
        weighted_contribution=_weighted(score, weight),
        comment=comment,
        details={"num_breaths": len(breath_durs), "overlong_breaths": len(bad_breaths)},
    )


def _score_overall_similarity(
    alignment: AlignmentResult,
) -> AttributeScore:
    """Global acoustic + prosodic similarity (DTW based)."""
    dtw = alignment.dtw_distance
    thresh_ex = METRIC_CONFIG.dtw_excellent_threshold
    thresh_good = METRIC_CONFIG.dtw_good_threshold

    if dtw <= thresh_ex:
        score = 93.0
    elif dtw <= thresh_good:
        score = 78.0 + (thresh_good - dtw) / (thresh_good - thresh_ex) * 15
    else:
        score = max(42.0, 78.0 - (dtw - thresh_good) * 48)

    comment = f"Global DTW distance = {dtw:.3f} (lower = more similar to reference)."
    if dtw > thresh_good:
        comment += " Consider listening to the reference again and matching its pacing and tone more closely."

    weight = get_normalized_weights()["overall_similarity"]
    return AttributeScore(
        name="Overall Similarity",
        score=round(score, 1),
        weighted_contribution=_weighted(score, weight),
        comment=comment,
        details={"dtw_distance": round(dtw, 4)},
    )


# =============================================================================
# MAIN EVALUATION ENTRYPOINT
# =============================================================================

def evaluate_recitation(
    mantra_id: str,
    user_audio: np.ndarray,
    user_sr: int,
    user_audio_path: Optional[Path] = None,
    asr_transcript: Optional[str] = None,
    play_reference: bool = False,
) -> EvaluationResult:
    """
    Full evaluation pipeline.
    Returns rich EvaluationResult with per-attribute scores + overall.
    """
    mantra = load_mantra(mantra_id)
    ref_path = mantra.reference_audio_path

    if not ref_path.exists():
        from core.audio_processor import ensure_reference_audio
        fallback = ensure_reference_audio(mantra_id)
        if fallback is not None:
            ref_path = fallback

    if play_reference:
        if ref_path and ref_path.exists():
            print("🔊 Playing reference...")
            ref_y, ref_sr = load_audio(ref_path)
            play_audio(ref_y, ref_sr)
        else:
            print("[yellow]No reference audio available to play.[/yellow]")

    # Align
    alignment = align_reference_and_user(ref_path, user_audio, user_sr)

    # Compute all attributes
    weights = get_normalized_weights()

    attr_scores: Dict[str, AttributeScore] = {}

    attr_scores["pronunciation_accuracy"] = _score_pronunciation_accuracy(
        alignment, mantra, asr_transcript
    )
    attr_scores["vowel_length_accuracy"] = _score_vowel_length(alignment, mantra)
    attr_scores["tone_pitch_stability"] = _score_pitch_stability(alignment)
    attr_scores["speed_tempo"] = _score_tempo(alignment)
    attr_scores["smoothness_fluency"] = _score_smoothness_fluency(alignment)
    attr_scores["volume_consistency"] = _score_volume_consistency(alignment)
    attr_scores["breath_control"] = _score_breath_control(alignment)
    attr_scores["overall_similarity"] = _score_overall_similarity(alignment)

    # Overall weighted score
    overall = sum(a.weighted_contribution for a in attr_scores.values())
    overall = round(min(98.5, max(28.0, overall)), 1)  # clamp for realism

    # Generate high-level summary + strengths/weaknesses
    sorted_attrs = sorted(attr_scores.values(), key=lambda x: x.score, reverse=True)
    strengths = [f"{a.name} ({a.score:.0f})" for a in sorted_attrs[:3] if a.score >= 74]
    weaknesses = [f"{a.name} ({a.score:.0f})" for a in sorted_attrs[-2:] if a.score < 68]

    summary = (
        f"Overall pronunciation score: {overall:.1f}/100 — {_label(overall)}. "
        f"Strongest area: {sorted_attrs[0].name}. "
        f"Focus area: {sorted_attrs[-1].name}."
    )

    # Sanskrit-specific tips (pull from mantra + generic)
    tips = mantra.guidance[:3]
    if attr_scores["vowel_length_accuracy"].score < 68:
        tips.append("Focus on holding long vowels (ī, ā, ū, e) for nearly twice the duration of short vowels.")
    if attr_scores["tone_pitch_stability"].score < 65:
        tips.append("Practice sustaining the oṃ and dhī on a steady pitch before adding the full mantra.")

    ts = time.strftime("%Y-%m-%d %H:%M:%S")

    result = EvaluationResult(
        mantra_id=mantra.id,
        mantra_name=mantra.name,
        timestamp=ts,
        user_audio_path=str(user_audio_path) if user_audio_path else None,
        reference_audio_path=str(ref_path),
        overall_score=overall,
        attribute_scores=attr_scores,
        summary=summary,
        strengths=strengths or ["Good effort — keep practicing!"],
        weaknesses=weaknesses or ["No major weaknesses detected"],
        sanskrit_tips=tips,
        alignment={
            "dtw_distance": round(alignment.dtw_distance, 4),
            "ref_duration": round(alignment.ref_duration, 2),
            "user_duration": round(alignment.user_duration, 2),
        },
        raw_measurements={
            "ref_prosody": {k: v for k, v in alignment.ref_features.items() if not isinstance(v, (dict, np.ndarray))},
            "user_prosody": {k: v for k, v in alignment.user_features.items() if not isinstance(v, (dict, np.ndarray))},
        },
    )
    return result
