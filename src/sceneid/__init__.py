"""Clip-to-video scene identification."""

import os

# transformers imports TensorFlow whenever it is installed, which adds ~20 s to startup
# and is never used here.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

__version__ = "2.0.0.dev0"
