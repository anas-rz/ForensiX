import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from graph_synth import sequence as synth_sequence

from forensix.physics.graph import train as G


def test_physdf_graph_prior_learning_loss_reduction():
    """Verify that physdf masked graph modeling transformer learns physical couplings and reduces reconstruction loss."""
    torch.manual_seed(0)

    seqs = [synth_sequence(seconds=6.0, seed=s) for s in range(30)]
    val_seqs = [synth_sequence(seconds=6.0, seed=100 + s) for s in range(10)]

    # Train graph prior and track loss history
    model, norm, history = G.train(
        seqs,
        steps=200,
        batch=16,
        dim=32,
        layers=3,
        lr=3e-3,
        log_every=200,
        log=lambda m: None,
    )

    # Verify that training loss is recorded
    assert isinstance(history, list)
    assert len(history) > 0
    assert history[-1] < 0.60, f"Final masked modeling loss too high: {history[-1]:.4f}"

    # Verify that calibration succeeds on held-out sequences
    calibration = G.calibrate(model, norm, val_seqs)
    assert "audio_video" in calibration
    assert "graph_av_mouth_from_audio" in calibration["audio_video"]
