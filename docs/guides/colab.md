# Working in Google Colab

Training and large evaluations run on Colab GPUs. This guide sets up a workflow where the repo
lives on Google Drive, survives disconnects, and syncs with GitHub.

## Which GPU

| GPU | VRAM | Use for |
|---|---|---|
| T4 | 16 GB | Small evaluations; works for everything, but slowest |
| **L4** | 24 GB | **Default.** Evaluations and most training; good speed per compute unit |
| A100 | 40/80 GB | Long training runs (`physdf`, `avid`, `id-motion` at full size) |

Video evaluation is limited by CPU work (decoding, face detection), not by the GPU, so an A100
barely beats an L4 there. The SDFVD benchmark ran 2–4 processes at once on one L4 to keep it
busy.

## One-time setup

1. **GitHub token:** create a fine-grained personal access token with access to only this
   repository, with **Contents: Read and write**.
2. **Colab secret:** in Colab, open 🔑 Secrets, add the token (e.g. as `GITHUB_TOKEN_FORENSIX`),
   and turn on notebook access for it.

## Each session

```python
# 1. Drive, secrets, cache location
from google.colab import drive, userdata
import os
drive.mount('/content/drive')
tok = userdata.get('GITHUB_TOKEN_FORENSIX').strip()
os.environ['GITHUB_TOKEN_FORENSIX'] = tok
os.environ['FORENSIX_CACHE'] = '/content/drive/MyDrive/forensix_cache'   # weights survive sessions

# credentials for git in this runtime only (the runtime's disk is wiped when it ends)
path = os.path.expanduser('~/.git-credentials')
with open(path, 'w') as f:
    f.write(f'https://x-access-token:{tok}@github.com\n')
os.chmod(path, 0o600)
```

```bash
%%bash
set -e
git config --global credential.helper store
git config --global user.name  "Your Name"
git config --global user.email "you@example.com"
cd /content/drive/MyDrive
[ -d ForensiX ] || git clone https://github.com/anas-rz/ForensiX.git
cd ForensiX
git config core.fileMode false     # Drive changes file permissions; without this every file shows as modified
git pull
pip install -q -e .
```

Push from the notebook after reviewing your changes:

```bash
%%bash
cd /content/drive/MyDrive/ForensiX
git status && git add -A && git commit -m "Describe the change" && git push
```

## Where to put data

| What | Where | Why |
|---|---|---|
| Repo | `/content/drive/MyDrive/ForensiX` | Survives disconnects; syncs with GitHub |
| Weights cache | `/content/drive/MyDrive/forensix_cache` | ~18 GB, downloaded once |
| Feature caches from `extract` | Drive | The expensive step; resumable |
| Datasets | `/content/` (local disk) | Drive is slow for many small reads; download again each session |
| Results | `results/` in the repo on Drive | Saved after every model; committed with the write-up |

## Common errors

| Error | Cause |
|---|---|
| `could not read Username for 'https://github.com'` | The repo is private and git has no credentials: run the credentials cell. |
| `Invalid username or token` | The token is empty, wrong or expired. The secret name in `userdata.get` must match the one in 🔑 Secrets exactly. |
| HTTP 404 from the GitHub API with a valid token | The token wasn't given access to this repository. |
| Every file shows as modified | Run `git config core.fileMode false`. |
| A run stops when the session drops | Rerun the same command: evaluation and `extract` steps resume. |

## Claude Code in Colab

Claude Code can run in Colab's terminal, next to the GPU:

```bash
curl -fsSL https://claude.ai/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"
cd /content/drive/MyDrive/ForensiX && claude      # log in with /login each session
```

Its session starts without your local context, so give it a self-contained prompt: the goal,
the paths, the exact commands, and any rules, e.g. "write results to files, never commit".
Its settings live on the runtime's disk and are wiped when the runtime ends.
