from .base import Embedder, l2_normalize
from .tiny import TinyImageEmbedder

# name -> sentence-transformers model id
_CLIP_MODELS = {
    "clip-vit-b32": "clip-ViT-B-32",
    "clip-vit-l14": "clip-ViT-L-14",
}

EMBEDDERS = sorted([*_CLIP_MODELS, "tiny16"])


def create_embedder(name: str, device: str = "auto", batch_size: int = 32) -> Embedder:
    if name == "tiny16":
        return TinyImageEmbedder(16)
    if name in _CLIP_MODELS:
        from .clip import ClipEmbedder

        return ClipEmbedder(_CLIP_MODELS[name], name, device=device, batch_size=batch_size)
    raise ValueError(f"unknown embedder {name!r}; choose from {', '.join(EMBEDDERS)}")


__all__ = ["EMBEDDERS", "Embedder", "TinyImageEmbedder", "create_embedder", "l2_normalize"]
