"""How window-based audio-visual detectors turn their per-window scores into one clip score.

- ``"mean"``: the model's own clip score (upstream behaviour), e.g. the sync
  confidence averaged over the whole track.
- ``"max-window"``: the most suspicious window. Better for short local edits
  (a replaced word), which a mean over the clip dilutes; on LAV-DF this rule,
  chosen on dev, raised test AUC by 0.07-0.11 (av_benchmark.md).
"""

CLIP_SCORES = ("mean", "max-window")


def check_clip_score(clip_score: str) -> str:
    if clip_score not in CLIP_SCORES:
        raise ValueError(f"clip_score must be one of {CLIP_SCORES}, not {clip_score!r}")
    return clip_score
