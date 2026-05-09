import tensorflow as tf
from tensorflow import keras
from keras.applications import MobileNetV2
import matplotlib.pyplot as plt

# -----------------------------
# CONFIGURATION
# -----------------------------

IMAGE_SIZE = (224,224)
BATCH_SIZE = 32
EPOCHS = 15

DATASET_PATH = r"/home/ravjot/datasets/Pomegranate Diseases Dataset"

# -----------------------------
# DATA LOADING
# -----------------------------

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)

# -----------------------------
# DATA AUGMENTATION
# -----------------------------

data_augmentation = keras.Sequential([
    keras.layers.RandomRotation(0.2),
    keras.layers.RandomZoom(0.2),
    keras.layers.RandomFlip("horizontal"),
])

# -----------------------------
# PREPROCESSING
# -----------------------------

preprocess = keras.applications.mobilenet_v2.preprocess_input

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# -----------------------------
# MODEL ARCHITECTURE
# -----------------------------

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

# freeze base model
base_model.trainable = False

inputs = keras.Input(shape=(224,224,3))

x = data_augmentation(inputs)

x = preprocess(x)

x = base_model(x, training=False)

# GlobalAveragePooling layer
x = keras.layers.GlobalAveragePooling2D()(x)

# Dropout layer (0.3)
x = keras.layers.Dropout(0.3)(x)

# Dense layer (128 units)
x = keras.layers.Dense(128, activation="relu")(x)

# Second Dropout
x = keras.layers.Dropout(0.3)(x)

# Output layer
outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = keras.Model(inputs, outputs)

# -----------------------------
# COMPILE MODEL
# -----------------------------

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# CALLBACKS
# -----------------------------

callbacks = [

    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),

    keras.callbacks.ModelCheckpoint(
        "best_mobilenet_pomegranate.keras",
        monitor="val_accuracy",
        save_best_only=True
    )
]

# -----------------------------
# TRAIN MODEL
# -----------------------------

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# -----------------------------
# SAVE MODEL
# -----------------------------

model.save("mobilenet_pomegranate_model.keras")

# -----------------------------
# VISUALIZATION
# -----------------------------

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")
plt.legend(["Train","Validation"])

plt.subplot(1,2,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Loss")
plt.legend(["Train","Validation"])

plt.show()