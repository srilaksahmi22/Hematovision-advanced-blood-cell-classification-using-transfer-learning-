Here’s a solid foundation for building Hematovision, an advanced blood cell classification model using transfer learning in Python with TensorFlow/Keras. It uses a pretrained backbone (e.g., EfficientNet, ResNet) and fine-tunes it on your blood-cell dataset.


---

🧠 1. Setup & Dependencies

!pip install tensorflow matplotlib sklearn

⚡ 2. Imports & Config

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

🗂️ 3. Data Preparation

# Paths to your dataset folders
train_dir = '/path/to/train'
val_dir   = '/path/to/val'

# Data augmentation & scaling
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)
val_gen = val_datagen.flow_from_directory(
    val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)
NUM_CLASSES = len(train_gen.class_indices)


---

🛠️ 4. Build the Model (Transfer Learning)

base_model = tf.keras.applications.EfficientNetB0(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # freeze

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


---

🚀 5. Initial Training

initial_epochs = 5
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=initial_epochs
)


---

🔄 6. Fine-Tuning

# Unfreeze some layers for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-10]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

fine_epochs = 10
total_epochs = initial_epochs + fine_epochs

history_fine = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1]
)


---

📊 7. Evaluate & Plot Results

def plot_training(h1, h2):
    acc = h1.history['accuracy'] + h2.history['accuracy']
    val_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']
    loss = h1.history['loss'] + h2.history['loss']
    val_loss = h1.history['val_loss'] + h2.history['val_loss']
    epochs = range(len(acc))

    plt.figure(figsize=(14,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, acc, label='train acc')
    plt.plot(epochs, val_acc, label='val acc')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1,2,2)
    plt.plot(epochs, loss, label='train loss')
    plt.plot(epochs, val_loss, label='val loss')
    plt.legend()
    plt.title('Loss')
    plt.show()

plot_training(history, history_fine)


---

🧪 8. Inference on New Images

def classify(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr)
    idx = np.argmax(preds, axis=1)[0]
    label = list(train_gen.class_indices.keys())[idx]
    print(f"Prediction: {label} ({preds[0][idx]*100:.2f}%)")

# Example:
classify('/path/to/test_image.png')


---

✅ Tips to Improve:

Backbone choice: Try other models like ResNet50, DenseNet121.

Augmentation: Increase dataset diversity with cutout, mixup, gamma adjustments.

Class imbalance: Use class_weight or focal loss.

Learning rate scheduling: e.g., ReduceLROnPlateau or Cosine decay.

Grad-CAM: Visualize model attention in predictions.



---

Let me know if you'd like help with:

Getting your dataset annotated and formatted

Switching to PyTorch implementation

Deploying the model (e.g., via TensorFlow Serving or converting to TensorFlow Lite)project executable files
