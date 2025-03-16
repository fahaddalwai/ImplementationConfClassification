import os
import shutil

# Paths
val_images_dir = 'IVLSR'
labels_file = 'IVLSR\imagenet_2012_validation_synset_labels.txt'

# Read labels
with open(labels_file, 'r') as f:
    labels = f.readlines()

labels = [label.strip() for label in labels]

# Create directories and move images
for i, label in enumerate(labels):
    img_filename = f'ILSVRC2012_val_{i+1:08d}.JPEG'
    src = os.path.join(val_images_dir, img_filename)
    dest_dir = os.path.join(val_images_dir, label)

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    dest = os.path.join(dest_dir, img_filename)
    shutil.move(src, dest)
