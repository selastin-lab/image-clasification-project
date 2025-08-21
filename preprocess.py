import tensorflow as tf

# Paths
train_dir = "dataset/train"
val_dir = "dataset/val"

# Parameters
img_size = (128, 128)  # resize all images to 128x128
batch_size = 32        # number of images per batch

# Load training dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=img_size,
    batch_size=batch_size
)

# Load validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=img_size,
    batch_size=batch_size
)

# Normalize pixel values (0–255 → 0–1)
normalization_layer = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

print("✅ Data preprocessing complete! Ready for training.")
