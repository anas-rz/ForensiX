"""Per-frame face geometry from MediaPipe FaceLandmarker (Apache-2.0): 478 3-D
landmarks, 52 ARKit blendshapes and the 4x4 head pose (translation in cm)."""

from dataclasses import dataclass

import numpy as np

from forensix.hub import WeightFile, fetch

FACE_LANDMARKER = WeightFile(
    url="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    sha256="64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    filename="physics/face_landmarker.task",
)


@dataclass
class FaceTrack:
    fps: float
    valid: np.ndarray  # (T,) bool: a face was found
    landmarks: np.ndarray  # (T, 478, 3) x, y in pixels, z in MediaPipe's relative depth units
    blendshapes: np.ndarray  # (T, 52) ARKit blendshape weights in [0, 1]
    blendshape_names: list[str]
    pose: np.ndarray  # (T, 4, 4) face-to-camera transform, translation in cm

    def __len__(self) -> int:
        return len(self.valid)


def track_face(frames: np.ndarray, fps: float) -> FaceTrack:
    """Run FaceLandmarker in video mode (temporal tracking) over RGB uint8 frames."""
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision

    options = vision.FaceLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=str(fetch(FACE_LANDMARKER))),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    t, h, w = frames.shape[:3]
    valid = np.zeros(t, bool)
    landmarks = np.full((t, 478, 3), np.nan, np.float32)
    blendshapes = np.full((t, 52), np.nan, np.float32)
    pose = np.full((t, 4, 4), np.nan, np.float32)
    names: list[str] = []
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        for i, frame in enumerate(frames):
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
            result = landmarker.detect_for_video(image, int(round(i * 1000 / fps)))
            if not result.face_landmarks:
                continue
            valid[i] = True
            landmarks[i] = [(p.x * w, p.y * h, p.z * w) for p in result.face_landmarks[0]]
            blendshapes[i] = [b.score for b in result.face_blendshapes[0]]
            names = names or [b.category_name for b in result.face_blendshapes[0]]
            pose[i] = np.asarray(result.facial_transformation_matrixes[0])
    return FaceTrack(fps, valid, landmarks, blendshapes, names, pose)
