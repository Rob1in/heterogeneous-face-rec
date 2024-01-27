#!/bin/bash

# Source folder
source_folder="/home/viam/face_recognition/datasets/clean/VIS"

# Destination folder
destination_folder="/home/viam/face_recognition/datasets/renamed_clean/VIS"

# Create a copy of the source folder
cp -r "$source_folder" "$destination_folder"

# Navigate to the destination folder
cd "$destination_folder"

# Rename files by removing prefix
for file in s?_*; do
    mv "$file" "${file#s?_}"
done
