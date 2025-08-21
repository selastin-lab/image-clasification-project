import os

# Base path where your project is located
base_path = r"C:\Users\ADMIN\image_classifier_project.selastin\dataset"

# Folder structure
folders = [
    "train/cat",
    "train/dog",
    "train/car",
    "train/chair",
    "val/cat",
    "val/dog",
    "val/car",
    "val/chair",
    "unknown"
]

# Create folders
for folder in folders:
    path = os.path.join(base_path, folder)
    os.makedirs(path, exist_ok=True)
    print(f"Created: {path}")

print("✅ All folders created successfully!")
