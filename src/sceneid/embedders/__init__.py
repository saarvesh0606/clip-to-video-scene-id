from .base import Embedder, l2_normalize
from .phash import PerceptualHashEmbedder
from .tiny import TinyImageEmbedder

# name -> sentence-transformers model id
_CLIP_MODELS = {
    "clip-vit-b32": "clip-ViT-B-32",
    "clip-vit-l14": "clip-ViT-L-14",
}

EMBEDDERS = sorted([*_CLIP_MODELS, "dinov2-base", "phash64", "sscd-disc-mixup", "tiny16"])


def create_embedder(name: str, device: str = "auto", batch_size: int = 32) -> Embedder:
    if name == "tiny16":
        return TinyImageEmbedder(16)
    if name == "phash64":
        return PerceptualHashEmbedder()
    if name in _CLIP_MODELS:
        from .clip import ClipEmbedder

        return ClipEmbedder(_CLIP_MODELS[name], name, device=device, batch_size=batch_size)
    if name == "sscd-disc-mixup":
        from .sscd import SSCDEmbedder

        return SSCDEmbedder(device=device, batch_size=min(batch_size, 64))
    if name == "dinov2-base":
        from .dinov2 import DinoV2Embedder

        return DinoV2Embedder(device=device, batch_size=min(batch_size, 64))
    raise ValueError(f"unknown embedder {name!r}; choose from {', '.join(EMBEDDERS)}")


__all__ = [
    "EMBEDDERS",
    "Embedder",
    "PerceptualHashEmbedder",
    "TinyImageEmbedder",
    "create_embedder",
    "l2_normalize",
]
