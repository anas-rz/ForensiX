"""The physdf graph: physical states as nodes, physical couplings as typed edges.

One graph per time step (25 steps/s). Nodes:
- face (MediaPipe + ICT-FaceKit): ``head`` pose, ``skull`` rigidity,
  ``jaw``, ``lips``, ``upper_face`` (eyes, brows);
- blood flow: one pulse node per skin region (rPPG);
- light: the face's spherical-harmonic lighting and the scene brightness;
- voice (source-filter): ``glottis`` (source), ``tract`` (filter: formants and
  spectral envelope), ``breath`` (energy envelope and pauses).

Edges carry the physics: the skull carries the face, muscles couple jaw, lips
and upper face; one heart drives every pulse region (coherence graph); the
source feeds the filter and the lungs power both; and the articulatory coupling
across modalities: the jaw and lips seen on video shape the vocal tract and the
energy heard in the audio.
"""

from forensix.physics.face.regions import REGIONS

STEP_S = 0.04  # 25 graph steps per second

# node -> (modality, feature names)
NODES: dict[str, tuple[str, list[str]]] = {
    "head": ("video", ["rot_x", "rot_y", "rot_z", "tx", "ty", "tz"]),
    "skull": ("video", ["rigid_residual"]),
    "jaw": ("video", ["jaw_open", "chin_drop", "jaw_lateral", "jaw_forward"]),
    "lips": ("video", ["aperture", "width", "funnel", "pucker", "close", "smile"]),
    "upper_face": ("video", ["blink_l", "blink_r", "brow_up", "brow_down"]),
    **{f"pulse_{r}": ("video", ["pulse", "dlog_green"]) for r in REGIONS},
    "light": ("video", ["log_ambient", "dir_x", "dir_y", "dir_z", "residual", "log_scene"]),
    "glottis": ("audio", ["log_f0", "voiced", "aperiodicity"]),
    "tract": ("audio", ["log_f1", "log_f2", "log_f3", "band_1", "band_2", "band_3", "band_4", "band_5", "band_6"]),
    "breath": ("audio", ["level_db", "speech", "since_pause_s"]),
}
NODE_NAMES = list(NODES)
PULSE_NODES = [f"pulse_{r}" for r in REGIONS]
VIDEO_NODES = [n for n in NODE_NAMES if NODES[n][0] == "video"]
AUDIO_NODES = [n for n in NODE_NAMES if NODES[n][0] == "audio"]
MOUTH_NODES = ["jaw", "lips"]

EDGE_TYPES = ["face", "pulse", "voice", "av"]
EDGES: list[tuple[str, str, str]] = [
    # skull and muscles
    ("head", "skull", "face"), ("head", "jaw", "face"), ("head", "upper_face", "face"), ("head", "light", "face"),
    ("skull", "jaw", "face"), ("skull", "upper_face", "face"),
    ("jaw", "lips", "face"), ("lips", "upper_face", "face"),
    # blood flow: one heart, every region; lighting and motion leak into skin colour
    *[(a, b, "pulse") for i, a in enumerate(PULSE_NODES) for b in PULSE_NODES[i + 1 :]],
    *[("light", p, "face") for p in PULSE_NODES],
    *[("head", p, "face") for p in PULSE_NODES],
    # speech production
    ("glottis", "tract", "voice"), ("glottis", "breath", "voice"), ("tract", "breath", "voice"),
    # articulatory coupling across modalities
    ("jaw", "tract", "av"), ("lips", "tract", "av"), ("jaw", "breath", "av"), ("lips", "breath", "av"),
    ("jaw", "glottis", "av"),
]

# residual groups scored as physdf constraints: (nodes whose residual counts,
# nodes hidden while predicting them; None = hide each node on its own, so it is
# predicted from its physical partners)
RESIDUAL_GROUPS: dict[str, tuple[list[str], list[str] | None]] = {
    "graph_skull_residual": (["head", "skull"], None),
    "graph_muscle_residual": (["jaw", "lips", "upper_face"], None),
    "graph_bloodflow_residual": (PULSE_NODES, None),  # each region from the others: coherence
    "graph_lighting_residual": (["light"], None),
    "graph_voice_residual": (AUDIO_NODES, None),
    # cross-modal: the mouth predicted without seeing it (from audio, head, upper face)
    "graph_av_mouth_from_audio": (MOUTH_NODES, MOUTH_NODES),
    # and the audio's articulation predicted from video alone
    "graph_av_audio_from_video": (["tract", "breath"], AUDIO_NODES),
}


def feature_dims() -> dict[str, int]:
    return {n: len(f) for n, (_, f) in NODES.items()}
