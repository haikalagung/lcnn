import os
import time
import numpy as np
import cv2
import tensorflow as tf

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# KONFIGURASI
# ============================================================

DATA_DIR = "/Volumes/OS/all video/Dataset CCTV"

CLASS_NAMES = [
    "Pelanggan Datang",
    "Pelanggan Menunggu",
    "Pelanggan Dilayani"
]

IMG_SIZE = 224
N_SPLITS = 5

# Untuk pengujian awal hanya 1 Fold
EPOCHS = 30
BATCH_SIZE = 32

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("LOAD DATASET")
print("=" * 60)

X = []
y = []

for label, class_name in enumerate(CLASS_NAMES):

    class_dir = os.path.join(DATA_DIR, class_name)

    files = sorted(os.listdir(class_dir))

    jumlah = 0

    for file_name in files:

        # Abaikan file AppleDouble macOS
        if file_name.startswith("._"):
            continue

        # Hanya file gambar
        if not file_name.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        file_path = os.path.join(class_dir, file_name)

        img = cv2.imread(file_path)

        if img is None:
            continue

        # Resize
        img = cv2.resize(
            img,
            (IMG_SIZE, IMG_SIZE)
        )

        # BGR -> RGB
        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        # Normalisasi
        img = img.astype("float32") / 255.0

        X.append(img)
        y.append(label)

        jumlah += 1

    print(
        f"{class_name:<25} : {jumlah} gambar"
    )


X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

print()
print("Shape X :", X.shape)
print("Shape y :", y.shape)

print()
print("Jumlah kelas:")

for i, class_name in enumerate(CLASS_NAMES):

    print(
        f"{class_name:<25} : {np.sum(y == i)}"
    )


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(

    [

        tf.keras.layers.RandomRotation(
            factor=0.015
        ),

        tf.keras.layers.RandomZoom(
            height_factor=(-0.10, 0.10),
            width_factor=(-0.10, 0.10)
        ),

        tf.keras.layers.RandomContrast(
            factor=0.10
        )

    ],

    name="data_augmentation"
)


# ============================================================
# CUSTOM LIGHTWEIGHT CNN V2
# ============================================================

def build_model():

    inputs = tf.keras.Input(
        shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    # --------------------------------------------------------
    # DATA AUGMENTATION
    # --------------------------------------------------------

    x = data_augmentation(inputs)

    # --------------------------------------------------------
    # INITIAL CONVOLUTION
    # --------------------------------------------------------

    x = tf.keras.layers.Conv2D(
        32,
        kernel_size=3,
        padding="same",
        use_bias=False
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.MaxPooling2D(
        pool_size=2
    )(x)


    # --------------------------------------------------------
    # DSC BLOCK 1
    # --------------------------------------------------------

    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=3,
        padding="same",
        use_bias=False
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(
        32,
        kernel_size=1,
        padding="same",
        use_bias=False
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.ReLU()(x)


    # --------------------------------------------------------
    # DSC BLOCK 2
    # --------------------------------------------------------

    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=3,
        padding="same",
        use_bias=False
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(
        64,
        kernel_size=1,
        padding="same",
        use_bias=False
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.MaxPooling2D(
        pool_size=2
    )(x)


    # --------------------------------------------------------
    # DSC BLOCK 3
    # --------------------------------------------------------

    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=3,
        padding="same",
        use_bias=False
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(
        128,
        kernel_size=1,
        padding="same",
        use_bias=False
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.MaxPooling2D(
        pool_size=2
    )(x)


    # --------------------------------------------------------
    # DSC BLOCK 4
    # --------------------------------------------------------

    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=3,
        padding="same",
        use_bias=False
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(
        256,
        kernel_size=1,
        padding="same",
        use_bias=False
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.ReLU()(x)


    # --------------------------------------------------------
    # DSC BLOCK 5
    # --------------------------------------------------------

    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=3,
        padding="same",
        use_bias=False
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(
        256,
        kernel_size=1,
        padding="same",
        use_bias=False
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.ReLU()(x)


    # --------------------------------------------------------
    # GLOBAL AVERAGE POOLING
    # --------------------------------------------------------

    x = tf.keras.layers.GlobalAveragePooling2D()(x)


    # --------------------------------------------------------
    # CLASSIFIER
    # --------------------------------------------------------

    x = tf.keras.layers.Dense(
        128,
        activation="relu"
    )(x)

    x = tf.keras.layers.Dropout(
        0.4
    )(x)

    outputs = tf.keras.layers.Dense(
        3,
        activation="softmax"
    )(x)


    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs
    )

    return model


# ============================================================
# STRATIFIED 5-FOLD
# ============================================================

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED
)


# ============================================================
# HANYA FOLD 1 UNTUK DIAGNOSTIK
# ============================================================

train_idx, val_idx = next(
    skf.split(X, y)
)

X_train = X[train_idx]
X_val = X[val_idx]

y_train = y[train_idx]
y_val = y[val_idx]


print()
print("=" * 60)
print("FOLD 1")
print("=" * 60)

print(
    "Training:",
    X_train.shape[0],
    "gambar"
)

print(
    "Validasi:",
    X_val.shape[0],
    "gambar"
)


# ============================================================
# CLASS WEIGHT
# ============================================================

kelas_unik = np.unique(y_train)

bobot_kelas = compute_class_weight(
    class_weight="balanced",
    classes=kelas_unik,
    y=y_train
)

class_weight = dict(
    zip(
        kelas_unik,
        bobot_kelas
    )
)


print()
print("CLASS WEIGHT")

for kelas, bobot in class_weight.items():

    print(
        f"{CLASS_NAMES[kelas]:<25} : {bobot:.4f}"
    )


# ============================================================
# MODEL
# ============================================================

model = build_model()


# ============================================================
# COMPILE
# ============================================================

optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0001
)

model.compile(

    optimizer=optimizer,

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print()
print("=" * 60)
print("MODEL SUMMARY")
print("=" * 60)

model.summary()


# ============================================================
# CALLBACK
# ============================================================

early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True,

    verbose=1
)


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=2,

    min_lr=1e-6,

    verbose=1
)


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 60)
print("TRAINING")
print("=" * 60)

start_training = time.time()


history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    class_weight=class_weight,

    callbacks=[
        early_stopping,
        reduce_lr
    ],

    verbose=1
)


training_time = (
    time.time()
    -
    start_training
)


# ============================================================
# PREDIKSI
# ============================================================

print()
print("=" * 60)
print("PREDIKSI")
print("=" * 60)

start_inference = time.time()


y_prob = model.predict(
    X_val,
    batch_size=BATCH_SIZE,
    verbose=0
)


inference_time = (
    time.time()
    -
    start_inference
)

y_pred = np.argmax(
    y_prob,
    axis=1
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_val,
    y_pred
)

precision_macro = precision_score(
    y_val,
    y_pred,
    average="macro",
    zero_division=0
)

recall_macro = recall_score(
    y_val,
    y_pred,
    average="macro",
    zero_division=0
)

f1_macro = f1_score(
    y_val,
    y_pred,
    average="macro",
    zero_division=0
)


# ============================================================
# FPS
# ============================================================

fps = (
    len(X_val)
    /
    inference_time
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_val,
    y_pred
)


# ============================================================
# HASIL
# ============================================================

print()
print("=" * 60)
print("EVALUASI")
print("=" * 60)

print()

print(
    f"Accuracy        : {accuracy:.4f}"
)

print(
    f"Precision Macro : {precision_macro:.4f}"
)

print(
    f"Recall Macro    : {recall_macro:.4f}"
)

print(
    f"F1 Macro        : {f1_macro:.4f}"
)

print(
    f"Training Time   : {training_time:.2f} detik"
)

print(
    f"Inference Time  : {inference_time:.4f} detik"
)

print(
    f"FPS             : {fps:.2f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("Confusion Matrix:")

print(cm)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("Classification Report:")

print(
    classification_report(
        y_val,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# BEST EPOCH
# ============================================================

best_val_loss_epoch = (
    np.argmin(
        history.history["val_loss"]
    )
    + 1
)

best_val_accuracy_epoch = (
    np.argmax(
        history.history["val_accuracy"]
    )
    + 1
)

best_val_loss = min(
    history.history["val_loss"]
)

best_val_accuracy = max(
    history.history["val_accuracy"]
)


print()
print("=" * 60)
print("BEST EPOCH")
print("=" * 60)

print(
    f"Best Val Loss       : {best_val_loss:.4f}"
)

print(
    f"Epoch Best Val Loss : {best_val_loss_epoch}"
)

print(
    f"Best Val Accuracy   : {best_val_accuracy:.4f}"
)

print(
    f"Epoch Best Val Acc  : {best_val_accuracy_epoch}"
)


print()
print("Selesai.")