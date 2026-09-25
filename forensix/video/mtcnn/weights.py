"""MTCNN weights, hash-checked copies of facenet-pytorch's data/*.pt."""

from pathlib import Path

from forensix.hub import WeightFile, fetch

_URL = (
    "https://raw.githubusercontent.com/timesler/facenet-pytorch/"
    "787da06156087cd6b616fe6608213722bddc30cd/data/{}.pt"
)
MTCNN_WEIGHTS = {
    "pnet": WeightFile(_URL.format("pnet"), "a2a71925e0b9996a42f63e47efc1ca19043e69558b5c523b978d611dfae49c8f", "mtcnn-pnet.pt"),
    "rnet": WeightFile(_URL.format("rnet"), "bbb937de72efc9ef83b186c49f5f558467a1d7e3453a8ece0d71a886633f6a86", "mtcnn-rnet.pt"),
    "onet": WeightFile(_URL.format("onet"), "165bfbe42940416ccfb977545cf0e976d5bf321f67083ae2aaaa5c764280118d", "mtcnn-onet.pt"),
}


def weight_path(net: str) -> Path:
    return fetch(MTCNN_WEIGHTS[net])
