from PIL import Image
import os

current_directory = os.getcwd()
input_dir = os.path.join(current_directory, "courtyard_dslr_undistorted",
                                "courtyard", "images", "dslr_images_undistorted")
output_dir = os.path.join(current_directory, "courtyard_dslr_undistorted",
                                "courtyard", "images", "new")
# Paths

target_size = (1552, 1033)

# Create output folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Resize each image
for filename in os.listdir(input_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path)
        resized_img = img.resize(target_size, Image.LANCZOS)
        resized_img.save(os.path.join(output_dir, filename))

print("✅ All images resized to 1552x1033.")
