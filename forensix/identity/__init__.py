"""Identity-based (reference-based) detectors: enroll a person from reference
media, then check whether new media really shows that person."""

from forensix.identity import face, motion  # noqa: F401  (registers models)
from forensix.identity.gallery import Gallery  # noqa: F401
