import tensorflow as tf
from tensorflow.keras import layers, models


# ============================================================
# 1. MEMBUAT MODEL LIGHTWEIGHT CNN
# ============================================================

model = models.Sequential([

    # Input
    layers.Input(shape=(224, 224, 3)),

    # ========================================================
    # Lightweight Block 1
    # Depthwise Convolution
    # ========================================================
    layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        padding="same"
    ),

    # Pointwise Convolution
    layers.Conv2D(
        filters=32,
        kernel_size=(1, 1),
        padding="same"
    ),

    # Aktivasi
    layers.ReLU(),

    # Pooling
    layers.MaxPooling2D(
        pool_size=(2, 2)
    ),


    # ========================================================
    # Lightweight Block 2
    # ========================================================
    layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        padding="same"
    ),

    # Pointwise Convolution
    layers.Conv2D(
        filters=64,
        kernel_size=(1, 1),
        padding="same"
    ),

    # Aktivasi
    layers.ReLU(),

    # Pooling
    layers.MaxPooling2D(
        pool_size=(2, 2)
    ),


    # ========================================================
    # Global Average Pooling
    # ========================================================
    layers.GlobalAveragePooling2D(),


    # ========================================================
    # Output
    # ========================================================
    layers.Dense(3),
    layers.Softmax()
])


# Menampilkan struktur model
model.summary()