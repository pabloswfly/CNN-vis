#!/usr/bin/env python3

import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} file.h5", file=sys.stderr)
        exit(1)

    model_fn = sys.argv[1]
    model = load_model(model_fn, custom_objects={"tf": tf}, compile=False)
