import numpy as np


def compute_eer(fake_scores: np.ndarray, real_scores: np.ndarray) -> tuple[float, float]:
    """Equal error rate and its threshold, where higher score = more likely fake.

    Returns ``(eer, threshold)`` with ``eer`` in [0, 1]. Follows the ASVspoof
    evaluation code (bona fide as the target class) so numbers are directly
    comparable with published results, including tie-breaking.
    """
    # ASVspoof convention: target = bona fide, higher score = more likely bona fide
    target = -np.asarray(real_scores, dtype=np.float64)
    nontarget = -np.asarray(fake_scores, dtype=np.float64)
    if target.size == 0 or nontarget.size == 0:
        raise ValueError("Need at least one fake and one real score")

    scores = np.concatenate([target, nontarget])
    labels = np.concatenate([np.ones(target.size), np.zeros(nontarget.size)])
    order = np.argsort(scores, kind="mergesort")
    labels = labels[order]

    tar_below = np.cumsum(labels)
    nontar_above = nontarget.size - (np.arange(1, scores.size + 1) - tar_below)
    frr = np.concatenate([[0], tar_below / target.size])
    far = np.concatenate([[1], nontar_above / nontarget.size])
    thresholds = np.concatenate([[scores[order[0]] - 0.001], scores[order]])

    i = int(np.argmin(np.abs(frr - far)))
    return float((frr[i] + far[i]) / 2), float(-thresholds[i])
