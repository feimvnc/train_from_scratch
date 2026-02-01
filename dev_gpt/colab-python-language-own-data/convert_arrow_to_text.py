"""
Convert Arrow dataset to plain text files.
This script reads the local dataset and saves each code sample to a separate .py file.
"""

import os
from pathlib import Path

from datasets import load_from_disk

# Configuration
dataset_dir = "local_dataset"
output_dir = "dataset_text_files"
output_format = "single"  # 'single' for one file, 'multiple' for separate files

print(f"Loading dataset from {dataset_dir}...")
dataset = load_from_disk(dataset_dir)
print(f"Loaded {len(dataset)} samples")

# Create output directory
os.makedirs(output_dir, exist_ok=True)

if output_format == "single":
    # Save all code to a single text file
    output_file = os.path.join(output_dir, "all_code.txt")
    print(f"\nSaving all code to {output_file}...")

    with open(output_file, "w", encoding="utf-8") as f:
        for i, item in enumerate(dataset):
            # Write separator with metadata
            f.write(f"\n{'='*80}\n")
            f.write(f"FILE {i+1}/{len(dataset)}: {item['repo_name']}/{item['path']}\n")
            f.write(f"License: {item['license']} | Size: {item['size']} bytes\n")
            f.write(f"{'='*80}\n\n")

            # Write the code content
            f.write(item["content"])
            f.write("\n\n")

            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1} samples...")

    file_size_mb = os.path.getsize(output_file) / 1024 / 1024
    print(f"\n✓ All code saved to: {output_file}")
    print(f"✓ File size: {file_size_mb:.2f} MB")

elif output_format == "multiple":
    # Save each sample to a separate file
    print(f"\nSaving code to separate files in {output_dir}...")

    for i, item in enumerate(dataset):
        # Create safe filename
        filename = (
            f"{i:05d}_{item['repo_name'].replace('/', '_')}_{Path(item['path']).name}"
        )
        filepath = os.path.join(output_dir, filename)

        # Write code to file
        with open(filepath, "w", encoding="utf-8") as f:
            # Add header comment
            f.write(f"# Source: {item['repo_name']}/{item['path']}\n")
            f.write(f"# License: {item['license']}\n")
            f.write(f"# Size: {item['size']} bytes\n\n")
            f.write(item["content"])

        if (i + 1) % 1000 == 0:
            print(f"  Saved {i + 1} files...")

    print(f"\n✓ Saved {len(dataset)} files to: {output_dir}")

# Also save metadata as CSV
print("\nSaving metadata to CSV...")
import csv

metadata_file = os.path.join(output_dir, "metadata.csv")
with open(metadata_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "index",
            "repo_name",
            "path",
            "size",
            "license",
            "line_mean",
            "line_max",
        ],
    )
    writer.writeheader()
    for i, item in enumerate(dataset):
        writer.writerow(
            {
                "index": i,
                "repo_name": item["repo_name"],
                "path": item["path"],
                "size": item["size"],
                "license": item["license"],
                "line_mean": item["line_mean"],
                "line_max": item["line_max"],
            }
        )

print(f"✓ Metadata saved to: {metadata_file}")
print("\nDone!")
