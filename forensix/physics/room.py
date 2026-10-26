"""phys-room: a recording happens in one room, through one microphone.

- reverberation stability: blind RT60 (slow end of free-decay times) of
  consecutive stretches of the file agrees; splices from other rooms change it;
- noise-floor continuity: the background level (quietest moments of each
  pause, since pauses hold breaths) doesn't step up or down;
- microphone self-noise: real microphones and preamps are never digitally
  silent; synthetic or post-processed speech often is;
- channel stability: the level-normalized long-term speech spectrum (microphone
  and codec colouration) keeps its shape across chunks;
- noise-spectrum stability: the background noise keeps its spectral shape
  (no switch between two noise regimes).

Priors are provisional ranges (marked in ``details``) until learned priors,
fitted on real media with ``scripts/calibrate_physics.py``, are loaded.
"""

import numpy as np

from forensix.physics.base import PhysicsAudioDetector
from forensix.physics.signals import room as R
from forensix.registry import ModelSpec, register

MIN_SECONDS = 6.0
CHUNK_S = 3.0
DIGITAL_SILENCE_DB = -90.0

PRIORS = {
    "rt60_variation": (0.35, 0.8),  # coefficient of variation across time groups
    # quietest-pause floor before vs after any point; real read speech (breathy,
    # short pauses) reaches ~10 dB
    "noise_floor_jump_db": (10.0, 20.0),
    "digital_silence_fraction": (0.2, 0.8),  # of pause frames
    # p90 RMS deviation of 3 s speech spectra; real read speech varies 2-4.6 dB
    "channel_variation_db": (6.0, 10.0),
    # RMS change of the median quiet-pause noise spectrum before vs after any point
    "noise_spectrum_change_db": (7.0, 12.0),  # real read speech (MP3) reaches ~6 dB
}


class RoomDetector(PhysicsAudioDetector):
    name = "phys-room"
    provisional_priors = PRIORS

    def analyze(self, x: np.ndarray, sr: int) -> dict:
        x = np.asarray(x, dtype=np.float64)
        if len(x) / sr < MIN_SECONDS:
            raise ValueError(f"Need at least {MIN_SECONDS:.0f} s of audio")
        levels = R.frame_levels_db(x, sr)
        speech = R.speech_mask(levels)
        hop = int(R.HOP_S * sr)
        pauses = R.pause_regions(speech, min_len=int(0.15 / R.HOP_S), trim=int(0.03 / R.HOP_S))

        # noise floor per pause: its quietest moments (p10). Only pauses count:
        # a window full of speech has no floor, and pauses often hold breaths.
        floors = np.array([np.percentile(levels[a : b + 1], 10) for a, b in pauses if (b - a) * R.HOP_S >= 0.1])
        pause_frames = np.concatenate([levels[a : b + 1] for a, b in pauses]) if pauses else np.array([])
        silent = pause_frames < DIGITAL_SILENCE_DB
        # noise spectra only from pauses at the floor (not breaths)
        global_floor = float(np.percentile(levels, 5))
        quiet = [(a, b) for a, b in pauses
                 if (b - a) * R.HOP_S >= 0.1 and np.percentile(levels[a : b + 1], 50) < global_floor + 6
                 and np.percentile(levels[a : b + 1], 50) > DIGITAL_SILENCE_DB]
        noise_spectra = [R.band_spectrum_db(x[a * hop : (b + 1) * hop + int(R.FRAME_S * sr)], sr) for a, b in quiet]
        pause_levels = [float(np.median(levels[a : b + 1])) for a, b in pauses]

        # per-chunk speech spectra (channel colouration)
        chunk = int(CHUNK_S * sr)
        noise_floor = global_floor
        speech_spectra = []
        for start in range(0, len(x) - chunk + 1, chunk):
            f0, f1 = start // hop, (start + chunk) // hop
            mask = np.repeat(speech[f0:f1], hop)[:chunk]
            mask = np.pad(mask, (0, chunk - len(mask)))  # the file's last samples have no frame level
            seg = x[start : start + chunk][mask] if mask.any() else np.array([])
            if len(seg) > sr // 2:
                speech_spectra.append(R.band_spectrum_db(seg, sr))

        # reverberation: free decays across the file, grouped by time
        decay_times, _ = R.decay_times_s(levels, noise_db=noise_floor)
        rt60 = R.rt60_by_time(decay_times)

        # step in the lower envelope: quietest pause floor before vs after each
        # point (>= 2 pauses per side). Breaths only raise a pause's level, so the
        # quietest pause is the one that shows the microphone/room noise.
        noise_jumps = np.array([abs(floors[:i].min() - floors[i:].min()) for i in range(2, len(floors) - 1)])
        measured = {
            "rt60_variation": float(rt60.std() / rt60.mean()) if len(rt60) >= 3 else float("nan"),
            "noise_floor_jump_db": float(noise_jumps.max()) if len(noise_jumps) else float("nan"),
            "digital_silence_fraction": float(silent.mean()) if len(pause_frames) else float("nan"),
            "channel_variation_db": R.shape_variation_db(speech_spectra),
            "noise_spectrum_change_db": R.shape_change_db(noise_spectra),
        }
        score, evidence = self.report(measured)
        return {
            "score": score,
            "constraints": evidence,
            "room": {
                "rt60_s_median": None if not len(rt60) else round(float(np.median(rt60)), 3),
                "noise_floor_dbfs": round(noise_floor, 1),
                "n_pauses": len(pauses),
            },
            "timeline": {
                "rt60_s_per_group": np.round(rt60, 3).tolist(),
                "noise_floor_dbfs_per_pause": np.round(floors, 1).tolist(),
                "pause_median_dbfs": np.round(pause_levels, 1).tolist(),
                "pause_start_s": [round(a * R.HOP_S, 2) for a, _ in pauses],
            },
        }


register(
    ModelSpec(
        name="phys-room",
        family="physics",
        builder=lambda weights_path, device="cpu": RoomDetector(),
        weights=None,
        license="ForensiX (algorithmic)",
        source="",
        paper="",
        description="Recording-environment physics (physdf): blind reverberation-time stability, "
        "noise-floor continuity, microphone self-noise (no digital silence), and stability of "
        "the channel and noise spectra. No model weights. Anomaly score in [0, 1] with "
        "per-constraint evidence and timelines. Priors: learned from real media when available, else provisional.",
    )
)
