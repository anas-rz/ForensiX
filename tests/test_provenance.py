import datetime
import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image, PngImagePlugin

import forensix
from forensix.provenance import check
from forensix.provenance.imwatermark.maxdct import EmbedMaxDct
from forensix.provenance.watermark_image import KNOWN_MESSAGES, decode, match

FACE = "tests/data/face_public_domain.jpg"
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def details(model, path, **options):
    return forensix.load(model, **options).predict(str(path)).details


# -- image watermark ------------------------------------------------------------------------

def test_watermark_decoder_matches_the_reference_bit_for_bit():
    rng = np.random.default_rng(0)
    images = [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for h, w in ((257, 391), (512, 512), (96, 160))]
    for img in images:
        for length in (32, 48, 136, 7):
            ref = EmbedMaxDct(wmLen=length, scales=[0, 36, 36], block=4).decode(img).astype(np.uint8)
            assert np.array_equal(ref, decode(img, length))


def embed(bgr, bits):
    out = EmbedMaxDct(watermarks=list(bits), wmLen=len(bits), scales=[0, 36, 36], block=4).encode(bgr.copy())
    return np.clip(out, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("generator", list(KNOWN_MESSAGES))
def test_generator_watermarks_are_found_and_named(tmp_path, generator):
    rgb = np.array(Image.open(FACE).convert("RGB").resize((512, 512)))
    marked = embed(np.ascontiguousarray(rgb[:, :, ::-1]), KNOWN_MESSAGES[generator])[:, :, ::-1]
    Image.fromarray(marked).save(tmp_path / "gen.png")
    d = details("watermark-image", tmp_path / "gen.png")
    assert (d["verdict"], d["label"], d["generator"]) == ("watermark_found", "fake", generator)
    assert d["bit_accuracy"][generator] == 1.0
    # ordinary JPEG (4:2:0 chroma) removes it: absence proves nothing
    Image.fromarray(marked).save(tmp_path / "gen.jpg", quality=95)
    assert details("watermark-image", tmp_path / "gen.jpg")["verdict"] == "no_watermark"


def test_clean_photo_has_no_watermark():
    d = details("watermark-image", FACE)
    assert d["verdict"] == "no_watermark" and d["label"] is None
    assert max(match(np.array(Image.open(FACE).convert("RGB"))[:, :, ::-1].copy()).values()) < 0.9


# -- C2PA --------------------------------------------------------------------------------------

def signing_chain(tmp_path):
    """A throwaway CA and ES256 signer, as C2PA expects (EKU emailProtection)."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    now = datetime.datetime.now(datetime.timezone.utc)
    name = lambda cn: x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn),  # noqa: E731
                                 x509.NameAttribute(NameOID.ORGANIZATION_NAME, "ForensiX Test")])
    ca_key, key = ec.generate_private_key(ec.SECP256R1()), ec.generate_private_key(ec.SECP256R1())
    ca = (x509.CertificateBuilder().subject_name(name("Test CA")).issuer_name(name("Test CA"))
          .public_key(ca_key.public_key()).serial_number(x509.random_serial_number())
          .not_valid_before(now - datetime.timedelta(days=1)).not_valid_after(now + datetime.timedelta(days=30))
          .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
          .add_extension(x509.KeyUsage(True, False, False, False, False, True, True, False, False), critical=True)
          .sign(ca_key, hashes.SHA256()))
    leaf = (x509.CertificateBuilder().subject_name(name("Test Signer")).issuer_name(ca.subject)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1)).not_valid_after(now + datetime.timedelta(days=30))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.KeyUsage(True, False, False, False, False, False, False, False, False), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.EMAIL_PROTECTION]), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
            .sign(ca_key, hashes.SHA256()))
    pem = serialization.Encoding.PEM
    (tmp_path / "ca.pem").write_bytes(ca.public_bytes(pem))
    return leaf.public_bytes(pem) + ca.public_bytes(pem), key.private_bytes(
        pem, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())


def c2pa_sign(src, dst, chain, key, source_type):
    c2pa = pytest.importorskip("c2pa")
    manifest = {"claim_generator_info": [{"name": "ForensiX test", "version": "1"}], "title": "test",
                "assertions": [{"label": "c2pa.actions", "data": {"actions": [
                    {"action": "c2pa.created",
                     "digitalSourceType": f"http://cv.iptc.org/newscodes/digitalsourcetype/{source_type}"}]}}]}
    signer = c2pa.Signer.from_info(c2pa.C2paSignerInfo(b"es256", chain, key, None))
    with c2pa.Builder(manifest) as builder:
        builder.sign_file(str(src), str(dst), signer)


@pytest.fixture
def signed(tmp_path):
    pytest.importorskip("c2pa")
    chain, key = signing_chain(tmp_path)
    c2pa_sign(FACE, tmp_path / "ai.jpg", chain, key, "trainedAlgorithmicMedia")
    c2pa_sign(FACE, tmp_path / "camera.jpg", chain, key, "digitalCapture")
    data = bytearray((tmp_path / "ai.jpg").read_bytes())
    data[-5000] ^= 0xFF
    (tmp_path / "tampered.jpg").write_bytes(bytes(data))
    return tmp_path


def test_c2pa_verdicts(signed):
    ai = details("c2pa", signed / "ai.jpg")
    assert (ai["verdict"], ai["label"]) == ("ai_generated", "fake") and "trainedAlgorithmicMedia" in ai["digital_source_types"]
    camera = details("c2pa", signed / "camera.jpg")
    assert (camera["verdict"], camera["label"]) == ("credentials_untrusted", None)   # anyone can sign "camera"
    tampered = details("c2pa", signed / "tampered.jpg")
    assert (tampered["verdict"], tampered["label"]) == ("tampered", "fake")
    assert "assertion.dataHash.mismatch" in tampered["failures"]
    assert details("c2pa", FACE)["verdict"] == "no_credentials"


def test_c2pa_trust_anchors_verify_capture(signed):
    camera = details("c2pa", signed / "camera.jpg", trust_anchors=str(signed / "ca.pem"))
    assert (camera["verdict"], camera["label"], camera["validation_state"]) == ("verified_capture", "real", "Trusted")


# -- metadata ----------------------------------------------------------------------------------

@pytest.fixture
def images(tmp_path):
    img = Image.open(FACE).convert("RGB").resize((512, 512))
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", "astronaut\nSteps: 30, Sampler: Euler a, CFG scale: 7, Seed: 1, Size: 512x512")
    img.save(tmp_path / "a1111.png", pnginfo=info)
    info = PngImagePlugin.PngInfo()
    info.add_text("prompt", '{"3": {"class_type": "KSampler"}}')
    img.save(tmp_path / "comfy.png", pnginfo=info)
    xmp = (b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
           b'<rdf:Description xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/" Iptc4xmpExt:'
           b'DigitalSourceType="http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"/>'
           b'</rdf:RDF></x:xmpmeta>')
    img.save(tmp_path / "iptc.jpg", xmp=xmp)
    for name, tags in (("photoshop", {0x0131: "Adobe Photoshop 25.0"}), ("mj", {0x0131: "Midjourney v6"}),
                       ("camera", {0x010F: "Canon", 0x0110: "Canon EOS R5"})):
        exif = Image.Exif()
        exif.update(tags)
        img.save(tmp_path / f"{name}.jpg", exif=exif)
    return tmp_path


@pytest.mark.parametrize("name,verdict,label,code", [
    ("a1111.png", "ai_generated", "fake", "meta.sd_parameters"),
    ("comfy.png", "ai_generated", "fake", "meta.comfyui"),
    ("iptc.jpg", "ai_generated", "fake", "meta.iptc_ai"),
    ("mj.jpg", "ai_generated", "fake", "meta.generator_tool"),
    ("photoshop.jpg", "edited", None, "meta.editor"),
    ("camera.jpg", "capture_metadata", None, "meta.camera"),
])
def test_image_metadata(images, name, verdict, label, code):
    d = details("metadata", images / name)
    assert (d["verdict"], d["label"]) == (verdict, label)
    assert code in {f["code"] for f in d["findings"]}


def test_metadata_never_vouches_for_authenticity(images):
    assert details("metadata", images / "camera.jpg")["label"] is None


@needs_ffmpeg
def test_video_metadata(tmp_path):
    src = tmp_path / "src.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=2",
                    "-f", "lavfi", "-i", "sine=duration=4", "-c:v", "libx264", "-c:a", "aac", "-metadata",
                    "comment=Generated with Runway Gen-3 Alpha", str(src)], check=True)
    d = details("metadata", src)
    assert (d["verdict"], d["label"]) == ("ai_generated", "fake")
    codes = {f["code"] for f in d["findings"]}
    assert {"meta.reencoded", "meta.track_length_mismatch"} <= codes      # FFmpeg; 2 s video vs 4 s audio


# -- AudioSeal -------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def audioseal_models():
    pytest.importorskip("audioseal")
    try:
        from audioseal import AudioSeal

        return AudioSeal.load_generator("audioseal_wm_16bits"), forensix.load("watermark-audio")
    except Exception as e:  # no network for the weights
        pytest.skip(f"AudioSeal weights unavailable: {e}")


def test_audioseal_detects_localizes_and_matches_upstream(audioseal_models):
    import torch
    from audioseal import AudioSeal

    from forensix.audio.io import load_audio

    generator, ours = audioseal_models
    speech = load_audio("tests/data/speech_cc0.flac", 16000)
    message = torch.tensor([[1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1]])
    with torch.no_grad():
        marked = generator(torch.from_numpy(speech)[None, None], sample_rate=16000, message=message)[0, 0].numpy()
    d = ours.predict(marked, sample_rate=16000).details
    assert (d["verdict"], d["message_bits"]) == ("watermark_found", "1011001011100011") and d["detection"] > 0.95
    assert ours.predict(speech, sample_rate=16000).details["verdict"] == "no_watermark"
    half = speech.copy()
    half[len(half) // 2:] = marked[len(half) // 2:]
    timeline = ours.predict(half, sample_rate=16000).details["timeline"]
    assert max(timeline[:4]) < 0.1 and min(timeline[-4:]) > 0.9
    upstream = AudioSeal.load_detector("audioseal_detector_16bits")
    x = torch.from_numpy(marked)[None, None]
    with torch.no_grad():
        assert torch.equal(upstream(x, sample_rate=16000)[0], ours.detector(x, sample_rate=16000)[0])


def test_restricted_unpickler_refuses_code(tmp_path):
    import os
    import pickle

    import torch

    from forensix.provenance.watermark_audio import load_checkpoint

    torch.save({"model": {"w": torch.zeros(1)}, "hook": os.getcwd}, tmp_path / "evil.pth")
    with pytest.raises(pickle.UnpicklingError, match="isn't allowed"):
        load_checkpoint(tmp_path / "evil.pth")
    torch.save({"model": {"w": torch.ones(2)}}, tmp_path / "ok.pth")
    assert torch.equal(load_checkpoint(tmp_path / "ok.pth")["w"], torch.ones(2))


# -- combined ---------------------------------------------------------------------------------

def test_check_combines_and_leads_with_the_decisive_verdict(images, signed):
    r = check(images / "a1111.png")
    assert (r["verdict"], r["label"], r["media"]) == ("ai_generated", "fake", "image")
    assert set(r["checks"]) == {"c2pa", "watermark-image", "metadata"}
    r = check(signed / "tampered.jpg")
    assert (r["verdict"], r["label"]) == ("tampered", "fake")
    r = check(FACE)
    assert (r["verdict"], r["label"]) == ("no_decisive_evidence", None)
