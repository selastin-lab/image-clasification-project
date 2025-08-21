import os
import shutil

# Path to your test folder
test_folder = "dataset/test"

# Create subfolders if they don't exist
os.makedirs(os.path.join(test_folder, "cat"), exist_ok=True)
os.makedirs(os.path.join(test_folder, "dog"), exist_ok=True)

# Loop through all files in test folder
for filename in os.listdir(test_folder):
    filepath = os.path.join(test_folder, filename)
    
    if os.path.isfile(filepath):
        # Move cat images
        if "cat" in filename.lower():
            shutil.move(filepath, os.path.join(test_folder, "cat", filename))
        # Move dog images
        elif "dog" in filename.lower():
            shutil.move(filepath, os.path.join(test_folder, "dog", filename))

print("Test images reorganized into cat/ and dog/ folders.")
