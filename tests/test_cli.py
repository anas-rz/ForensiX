import json

import pytest

from forensix.__main__ import main


def test_cli_prints_a_line_per_model(capsys):
    assert main(["tests/data/face_public_domain.jpg", "--models", "metadata,watermark-image"]) == 0
    out = capsys.readouterr().out
    assert "face_public_domain.jpg (image)" in out and "metadata" in out and "watermark-image" in out


def test_cli_json_and_errors(capsys):
    main(["tests/data/speech_cc0.flac", "--models", "metadata,avcons", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert report["media"] == "audio" and "verdict" in report["results"]["metadata"]
    assert "error" in report["results"]["avcons"]          # needs trained weights: reported, not raised
    with pytest.raises(SystemExit):
        main(["does-not-exist.jpg"])
