"""Physics-based detectors (physdf): violations of the physics of real faces and voices."""

from forensix.physics import bloodflow, lighting, motion, room, voice  # noqa: F401  (registers models)

DETECTORS = {
    "phys-bloodflow": bloodflow.BloodFlowDetector,
    "phys-motion": motion.MotionDetector,
    "phys-lighting": lighting.LightingDetector,
    "phys-voice": voice.VoiceDetector,
    "phys-room": room.RoomDetector,
}
VIDEO_DETECTORS = ("phys-bloodflow", "phys-motion", "phys-lighting")
AUDIO_DETECTORS = ("phys-voice", "phys-room")

from forensix.physics import physdf  # noqa: E402,F401  (registers physdf; uses DETECTORS)
