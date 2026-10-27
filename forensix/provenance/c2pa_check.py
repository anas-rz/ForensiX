"""C2PA Content Credentials: verify the signed provenance a file carries.

Uses the official c2pa-python SDK (MIT OR Apache-2.0, Content Authenticity
Initiative). Install with ``pip install -e ".[provenance]"``.

Verdicts, strongest first:

- ``tampered``: the credentials are signed, but the content no longer matches its
  signed hashes, so it was changed after signing (label ``fake``);
- ``ai_generated``: the manifest (or one of its ingredients) declares generative-AI
  content (IPTC digital source type ``trainedAlgorithmicMedia``, ``compositeWith...``,
  ``compositeSynthetic``) (label ``fake``);
- ``verified_capture``: credentials from a trusted signer declare a capture (camera,
  film scan) with no AI anywhere in the history (label ``real``);
- ``ai_enhanced``, ``credentials_untrusted``, ``trusted_credentials``,
  ``invalid_credentials``: evidence to read, but not decisive (no label);
- ``no_credentials``: nothing to verify, which is the common case and says nothing
  about authenticity.

Trust: a signature only proves who signed. Without trust anchors, every signer is
"untrusted", so capture claims can't be verified (anyone can sign "this is a photo").
Pass ``trust_anchors`` (a PEM bundle, e.g. the C2PA trust list) to verify signers.
AI declarations count whatever the signer, since nobody gains by falsely labelling
their own media as AI.
"""

import json
from pathlib import Path

from forensix.provenance.evidence import Finding, provenance_prediction
from forensix.registry import ModelSpec, register

IPTC = "http://cv.iptc.org/newscodes/digitalsourcetype/"
AI_SOURCES = {"trainedAlgorithmicMedia", "compositeWithTrainedAlgorithmicMedia", "compositeSynthetic",
              "trainedAlgorithmicData"}
ENHANCED_SOURCES = {"algorithmicallyEnhanced", "algorithmicMedia"}
CAPTURE_SOURCES = {"digitalCapture", "computationalCapture", "negativeFilm", "positiveFilm", "print",
                   "compositeCapture"}
# the content changed after signing: its hash no longer matches the signed one
TAMPER_CODES = {"assertion.dataHash.mismatch", "assertion.bmffHash.mismatch", "assertion.boxesHash.mismatch",
                "assertion.collectionHash.mismatch", "assertion.hashedURI.mismatch"}


class C2PAChecker:
    name = "c2pa"

    def __init__(self, trust_anchors: str | Path | None = None):
        try:
            import c2pa
        except ImportError as e:
            raise ImportError('c2pa needs the c2pa-python package: pip install -e ".[provenance]"') from e
        self._c2pa = c2pa
        self.trust_anchors = str(trust_anchors) if trust_anchors else None
        settings = None
        if trust_anchors:
            settings = {"trust": {"trust_anchors": Path(trust_anchors).read_text()}, "verify": {"verify_trust": True}}
        self._context = c2pa.Context.from_dict(settings) if settings else None

    def read(self, path: str | Path) -> dict | None:
        """The manifest store as a dict, or None when the file has no credentials."""
        kwargs = {"context": self._context} if self._context else {}
        try:
            reader = self._c2pa.Reader.try_create(str(path), **kwargs)
        except self._c2pa.C2paError as e:
            if "not supported" in str(e).lower() or "notsupported" in str(e).lower():
                return None
            raise
        if reader is None:
            return None
        with reader:
            return json.loads(reader.json())

    def predict(self, media) -> "Prediction":  # noqa: F821
        try:
            store = self.read(media)
        except self._c2pa.C2paError as e:
            return provenance_prediction(self.name, "invalid_credentials", None,
                                         "The file's C2PA data couldn't be read.",
                                         [Finding("c2pa.unreadable", "moderate", f"Unreadable C2PA data: {e}")])
        if store is None:
            return provenance_prediction(self.name, "no_credentials", None,
                                         "No Content Credentials (C2PA) in this file.", [])
        return self._judge(store)

    def _judge(self, store: dict) -> "Prediction":  # noqa: F821
        manifests = store.get("manifests", {})
        active = manifests.get(store.get("active_manifest"), {})
        state = store.get("validation_state")  # Invalid | Valid | Trusted
        results = (store.get("validation_results") or {}).get("activeManifest", {})
        failures = [f.get("code") for f in results.get("failure", [])]
        sig = active.get("signature_info", {})
        signer = sig.get("issuer") or sig.get("common_name") or "unknown signer"
        tools = [g.get("name") for g in active.get("claim_generator_info", []) if g.get("name")] or \
            ([active["claim_generator"]] if active.get("claim_generator") else [])
        sources, actions = _sources_and_actions(manifests)
        ai = sorted({s for s in sources if s in AI_SOURCES})
        enhanced = sorted({s for s in sources if s in ENHANCED_SOURCES})
        capture = sorted({s for s in sources if s in CAPTURE_SOURCES})
        trusted = state == "Trusted"
        extra = {"validation_state": state, "signer": signer, "signed_at": sig.get("time"),
                 "claim_generator": tools, "digital_source_types": sorted(set(sources)), "actions": actions,
                 "title": active.get("title"), "manifests": len(manifests), "failures": failures,
                 "trust_list": bool(self.trust_anchors)}
        findings = [Finding("c2pa.present", "info", f"Content Credentials signed by {signer} ({state or 'unknown'})"
                            + (f" with {', '.join(tools)}" if tools else ""), {"validation_state": state})]

        tampered = sorted(set(failures) & TAMPER_CODES)
        if tampered:
            findings.append(Finding("c2pa.tampered", "strong",
                                    "The content changed after it was signed: its hash no longer matches the "
                                    "signed one", {"failures": tampered}))
            return provenance_prediction(self.name, "tampered", "fake",
                                         f"Content Credentials signed by {signer} no longer match the content: "
                                         "it was modified after signing.", findings, **extra)
        if state == "Invalid":
            findings.append(Finding("c2pa.invalid", "moderate", "The credentials don't validate",
                                    {"failures": failures}))
            return provenance_prediction(self.name, "invalid_credentials", None,
                                         "The file has Content Credentials, but they don't validate.",
                                         findings, **extra)
        if ai:
            findings.append(Finding("c2pa.ai_declared", "strong",
                                    f"The credentials declare AI-generated content ({', '.join(ai)})",
                                    {"digital_source_types": ai, "signer_trusted": trusted}))
            return provenance_prediction(self.name, "ai_generated", "fake",
                                         f"Content Credentials from {signer} declare AI-generated content "
                                         f"({', '.join(ai)}).", findings, **extra)
        if enhanced:
            findings.append(Finding("c2pa.ai_enhanced", "moderate",
                                    f"The credentials declare algorithmic enhancement ({', '.join(enhanced)})",
                                    {"digital_source_types": enhanced}))
            return provenance_prediction(self.name, "ai_enhanced", None,
                                         "Content Credentials declare algorithmically enhanced content.",
                                         findings, **extra)
        if trusted and capture:
            findings.append(Finding("c2pa.verified_capture", "strong",
                                    f"A trusted signer declares a capture ({', '.join(capture)}), with no AI "
                                    "in its history", {"digital_source_types": capture}))
            return provenance_prediction(self.name, "verified_capture", "real",
                                         f"Verified Content Credentials from {signer} declare a capture with no AI "
                                         "in its history.", findings, **extra)
        if trusted:
            return provenance_prediction(self.name, "trusted_credentials", None,
                                         f"Verified Content Credentials from {signer}; they declare no AI, but no "
                                         "capture either.", findings, **extra)
        findings.append(Finding("c2pa.untrusted", "info",
                                "The signer isn't on the trust list, so its claims can't be verified"
                                + ("" if self.trust_anchors else " (no trust list configured)"),
                                {"declared_sources": capture}))
        return provenance_prediction(self.name, "credentials_untrusted", None,
                                     f"Content Credentials from an unverified signer ({signer})"
                                     + (f" declare {', '.join(capture)}" if capture else "") + "; the content is intact "
                                     "since signing.", findings, **extra)

    def predict_batch(self, inputs) -> list:
        return [self.predict(x) for x in inputs]


def _sources_and_actions(manifests: dict) -> tuple[list[str], list[str]]:
    """Every digital source type and action in the whole manifest store (all ingredients)."""
    sources, actions = [], []

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k in ("digitalSourceType", "Iptc4xmpExt:DigitalSourceType") and isinstance(v, str):
                    sources.append(v.rsplit("/", 1)[-1])
                elif k == "action" and isinstance(v, str):
                    actions.append(v)
                else:
                    walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(manifests)
    return sources, sorted(set(actions))


def _build(weights_path, device: str = "cpu", trust_anchors: str | None = None) -> C2PAChecker:
    import os

    return C2PAChecker(trust_anchors or os.environ.get("FORENSIX_C2PA_TRUST_ANCHORS"))


register(
    ModelSpec(
        name="c2pa",
        family="provenance",
        builder=_build,
        weights=None,
        license="MIT OR Apache-2.0 (c2pa-python, Content Authenticity Initiative). No model weights.",
        source="https://github.com/contentauth/c2pa-python",
        paper="https://c2pa.org/specifications/",
        description="Verifies C2PA Content Credentials: tampering after signing, declared AI generation "
        "(IPTC digital source types, including ingredients), trusted capture. Pass trust_anchors (PEM) to "
        "verify signers. Images, video and audio.",
    )
)
