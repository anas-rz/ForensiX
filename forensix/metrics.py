import numpy as np
from scipy.stats import rankdata


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


def _binary_scores(fake_scores, real_scores) -> tuple[np.ndarray, np.ndarray]:
    fake_scores = np.asarray(fake_scores, dtype=np.float64)
    real_scores = np.asarray(real_scores, dtype=np.float64)
    if fake_scores.size == 0 or real_scores.size == 0:
        raise ValueError("Need at least one fake and one real score")
    scores = np.concatenate([fake_scores, real_scores])
    is_fake = np.concatenate([np.ones(fake_scores.size), np.zeros(real_scores.size)])
    return scores, is_fake


def roc_auc(fake_scores, real_scores) -> float:
    """Area under the ROC curve (fake = positive), ties counted as 1/2."""
    scores, is_fake = _binary_scores(fake_scores, real_scores)
    # Mann-Whitney U with average ranks for ties
    ranks = rankdata(scores)
    n_fake = is_fake.sum()
    n_real = is_fake.size - n_fake
    u = ranks[is_fake == 1].sum() - n_fake * (n_fake + 1) / 2
    return float(u / (n_fake * n_real))


def average_precision(fake_scores, real_scores) -> float:
    """Average precision (fake = positive), same definition as scikit-learn:
    sum over thresholds of (recall_n - recall_{n-1}) * precision_n."""
    scores, is_fake = _binary_scores(fake_scores, real_scores)
    order = np.argsort(-scores, kind="mergesort")
    scores, is_fake = scores[order], is_fake[order]
    # one point per distinct threshold
    last = np.r_[np.diff(scores) != 0, True]
    tp = np.cumsum(is_fake)[last]
    fp = np.cumsum(1 - is_fake)[last]
    precision = tp / (tp + fp)
    recall = tp / is_fake.sum()
    return float(np.sum(np.diff(np.r_[0, recall]) * precision))


def binary_summary(fake_scores, real_scores, threshold: float = 0.5) -> dict:
    """ROC-AUC, AP and accuracies at ``threshold`` for fake-vs-real scores."""
    fake = np.asarray(fake_scores, dtype=np.float64)
    real = np.asarray(real_scores, dtype=np.float64)
    real_acc = float(np.mean(real < threshold))
    fake_acc = float(np.mean(fake >= threshold))
    return {
        "auc": round(roc_auc(fake, real), 4),
        "ap": round(average_precision(fake, real), 4),
        "balanced_acc": round((real_acc + fake_acc) / 2, 4),
        "real_acc": round(real_acc, 4),
        "fake_acc": round(fake_acc, 4),
        "n_fake": int(fake.size),
        "n_real": int(real.size),
    }
