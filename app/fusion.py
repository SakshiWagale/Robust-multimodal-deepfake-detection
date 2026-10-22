import os

# Configurable defaults (can be overridden via environment variables or function parameters)
DEFAULT_VIDEO_WEIGHT = float(os.getenv("DF_VIDEO_WEIGHT", "0.6"))
DEFAULT_AUDIO_WEIGHT = float(os.getenv("DF_AUDIO_WEIGHT", "0.4"))
DEFAULT_THRESHOLD = float(os.getenv("DF_THRESHOLD", "0.55"))


def fuse_predictions(video_score: float = None,
                     audio_score: float = None,
                     video_weight: float = DEFAULT_VIDEO_WEIGHT,
                     audio_weight: float = DEFAULT_AUDIO_WEIGHT,
                     threshold: float = DEFAULT_THRESHOLD) -> dict:
    """
    Performs multimodal decision-level fusion.
    Both video_score and audio_score represent probability of REAL in [0.0, 1.0].
    
    If both modalities are available:
        final_probability = (w_v * s_v + w_a * s_a) / (w_v + w_a)
    If only video is available (e.g. video without audio track):
        final_probability = video_score
    If only audio is available (e.g. audio upload):
        final_probability = audio_score
    
    Verdict:
        REAL if final_probability >= threshold else FAKE
    """
    if video_score is None and audio_score is None:
        raise ValueError("Cannot perform fusion: both video and audio scores are missing.")

    analysis_type = "unknown"

    if video_score is not None and audio_score is not None:
        # Normalize weights in case custom weights do not sum to 1.0
        total_weight = video_weight + audio_weight
        norm_v_weight = video_weight / total_weight
        norm_a_weight = audio_weight / total_weight

        final_score = (norm_v_weight * video_score) + (norm_a_weight * audio_score)
        analysis_type = "multimodal (video + audio)"
    elif video_score is not None:
        final_score = video_score
        analysis_type = "unimodal (video only)"
    else:
        final_score = audio_score
        analysis_type = "unimodal (audio only)"

    final_score = round(float(final_score), 4)

    # Multimodal Late-Fusion Decision Logic
    # 1. When both modalities are available, weighted average combines the streams:
    #    final_score = (w_v * s_v + w_a * s_a) / (w_v + w_a)
    # 2. If a modality detects a severe anomaly (e.g. video face fake < 0.45 or synthetic cloned voice < 0.30),
    #    it flags as FAKE.
    # 3. Overall, if final_score >= threshold -> REAL, else -> FAKE.
    if video_score is not None and audio_score is not None:
        if video_score < 0.45 or audio_score < 0.30 or final_score < threshold:
            verdict = "FAKE"
            confidence = round((1.0 - final_score) * 100, 2)
        else:
            verdict = "REAL"
            confidence = round(final_score * 100, 2)
    elif final_score >= threshold:
        verdict = "REAL"
        confidence = round(final_score * 100, 2)
    else:
        verdict = "FAKE"
        confidence = round((1.0 - final_score) * 100, 2)

    return {
        "video_score": round(video_score, 4) if video_score is not None else None,
        "audio_score": round(audio_score, 4) if audio_score is not None else None,
        "final_score": final_score,
        "verdict": verdict,
        "confidence": confidence,
        "threshold": threshold,
        "video_weight": video_weight,
        "audio_weight": audio_weight,
        "analysis_type": analysis_type
    }
