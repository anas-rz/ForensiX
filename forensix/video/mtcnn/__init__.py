"""MTCNN face detector (Zhang et al., 2016), vendored from facenet-pytorch (MIT).

The weights descend from the original MTCNN release (MIT) via
davidsandberg/facenet (MIT). facenet-pytorch itself isn't a dependency because
it pins old torch versions.
"""

from forensix.video.mtcnn.weights import MTCNN_WEIGHTS, weight_path

__all__ = ["MTCNN_WEIGHTS", "weight_path"]
# the detector itself (imports cv2): from forensix.video.mtcnn.mtcnn import MTCNN
