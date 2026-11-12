"""MkDocs hooks: put the top-level reference pages in the site and fix links that leave docs/.

The Markdown under docs/ is written for GitHub, where ``../README.md`` or ``../Dockerfile`` work.
In the built site those targets don't exist, so at build time:

- the repository-root pages in ROOT_PAGES (model reference, leaderboard, benchmark reports) are
  added to the site, and links to them are pointed at their pages;
- links to any other file outside docs/ (Dockerfile, scripts, deploy/) go to it on GitHub.

The sources stay unchanged, so both GitHub and the site keep working. Wired up in mkdocs.yml.

The browser studio (forensix/server/static/studio/) is published at studio/, with the web manager's
stylesheet it builds on. Its voice model is too big for the repository: when STUDIO_ECAPA points at
an exported ECAPA ONNX file (the docs workflow makes one), it's published at studio/models/ecapa.onnx.
"""

import os
import posixpath
import re
from pathlib import Path

from mkdocs.structure.files import File

ROOT = Path(__file__).resolve().parent.parent
REPO = "https://github.com/anas-rz/ForensiX"
BRANCH = "main"
ROOT_PAGES = {  # repository file -> page in the site
    "README.md": "reference/models.md",
    "LEADERBOARD.md": "reference/leaderboard.md",
    "benchmark.md": "reference/benchmark-video-image.md",
    "av_benchmark.md": "reference/benchmark-audiovisual.md",
    "identity_benchmark.md": "reference/benchmark-identity.md",
    "physics_benchmark.md": "reference/benchmark-physics.md",
    "person_anomaly_benchmark.md": "reference/benchmark-person-anomaly.md",
}
STATIC = ROOT / "forensix" / "server" / "static"
STUDIO_FILES = {"studio/index.html": "studio/index.html", "studio/engine.js": "studio/engine.js",
                "studio/studio.js": "studio/studio.js", "studio/studio.css": "studio/studio.css", "style.css": "style.css"}
SOURCE_OF = {page: repo_file for repo_file, page in ROOT_PAGES.items()}
LINK = re.compile(r"(!?\[[^\]]*\]\()([^)\s]+)(\))|(<(?:img\s[^>]*?src|source\s[^>]*?srcset)=\")([^\"]+)(\")")
FENCE = re.compile(r"^(```|~~~)")


def on_files(files, config):
    for repo_file, page in ROOT_PAGES.items():
        files.append(File.generated(config, page, content=(ROOT / repo_file).read_text(encoding="utf-8")))
    for src, dest in STUDIO_FILES.items():
        files.append(File.generated(config, dest, abs_src_path=str(STATIC / src)))
    ecapa = os.environ.get("STUDIO_ECAPA")
    if ecapa and Path(ecapa).exists():
        files.append(File.generated(config, "studio/models/ecapa.onnx", abs_src_path=str(Path(ecapa).resolve())))
    return files


def on_page_markdown(markdown, page, config, files):
    src = page.file.src_uri
    # where the page's source lives, relative to the repository root
    here = posixpath.dirname(SOURCE_OF.get(src) or "docs/" + src)
    # Markdown links are relative to the source file (MkDocs adjusts them); raw HTML is relative to the served URL
    md_base, html_base = posixpath.dirname(src) or ".", page.url.rstrip("/") if page.url.endswith("/") else posixpath.dirname(page.url)
    out, fenced = [], False
    for line in markdown.split("\n"):
        if FENCE.match(line.lstrip()):
            fenced = not fenced
        if not fenced:
            line = LINK.sub(lambda m: m[1] + _target(m[2], here, md_base) + m[3] if m[1]
                            else m[4] + _target(m[5], here, html_base or ".") + m[6], line)
        out.append(line)
    return "\n".join(out)


def _target(url: str, here: str, base: str) -> str:
    """``url`` as written in a page whose source is in ``here`` (repo-relative), made relative to ``base``
    (a docs-relative directory), or a GitHub URL when it points outside the site."""
    if re.match(r"^[a-z][a-z0-9+.-]*:|^#|^/", url):  # absolute URL, anchor or site-absolute path
        return url
    path, _, anchor = url.partition("#")
    resolved = posixpath.normpath(posixpath.join(here, path))
    suffix = "#" + anchor if anchor else ""
    if resolved in ROOT_PAGES:
        site_path = ROOT_PAGES[resolved]
    elif resolved == "docs" or resolved.startswith("docs/"):
        site_path = resolved[len("docs/"):] or "README.md"
    else:
        kind = "tree" if (ROOT / resolved).is_dir() else "blob"
        return f"{REPO}/{kind}/{BRANCH}/{resolved}{suffix}"
    return posixpath.relpath(site_path, base) + suffix
