"""Enrolled identities: reference embeddings per person, saved to one .npz file.

A gallery belongs to one embedding model (``model``); loading it into another
model is refused, because embeddings from different models aren't comparable.
"""

import json
from pathlib import Path

import numpy as np


class Gallery:
    def __init__(self, model: str):
        self.model = model
        self.people: dict[str, dict] = {}  # name -> {"embeddings": (N, D), "sources": [...]}

    def __contains__(self, name: str) -> bool:
        return name in self.people

    def __len__(self) -> int:
        return len(self.people)

    def names(self) -> list[str]:
        return sorted(self.people)

    def add(self, name: str, embeddings: np.ndarray, sources: list[str]) -> None:
        embeddings = np.asarray(embeddings, np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        if name in self.people:
            old = self.people[name]
            embeddings = np.concatenate([old["embeddings"], embeddings])
            sources = old["sources"] + list(sources)
        self.people[name] = {"embeddings": embeddings, "sources": list(sources)}

    def remove(self, name: str) -> None:
        self.people.pop(name)

    def embeddings(self, name: str) -> np.ndarray:
        if name not in self.people:
            raise KeyError(f"{name!r} is not enrolled. Enrolled: {', '.join(self.names()) or 'nobody'}")
        return self.people[name]["embeddings"]

    def save(self, path: str | Path) -> None:
        arrays = {f"emb_{i}": p["embeddings"] for i, p in enumerate(self.people.values())}
        meta = {"model": self.model, "people": [{"name": n, "sources": p["sources"]} for n, p in self.people.items()]}
        np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)

    @classmethod
    def load(cls, path: str | Path, model: str | None = None) -> "Gallery":
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        if model is not None and meta["model"] != model:
            raise ValueError(f"{path} holds {meta['model']!r} embeddings, not {model!r}")
        g = cls(meta["model"])
        for i, p in enumerate(meta["people"]):
            g.people[p["name"]] = {"embeddings": z[f"emb_{i}"], "sources": p["sources"]}
        return g


def top_k_similarity(queries: np.ndarray, references: np.ndarray, k: int = 3) -> np.ndarray:
    """(Q,) mean of each query's k highest cosine similarities to the references
    (robust to a few bad reference frames, less optimistic than the single max)."""
    q = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    sims = q @ references.T
    k = min(k, sims.shape[1])
    return np.sort(sims, axis=1)[:, -k:].mean(1)
