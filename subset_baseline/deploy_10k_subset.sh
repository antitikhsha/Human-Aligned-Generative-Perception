#!/bin/bash
# Deploy and run 10k balanced subset creation on AWS instance
# Usage: bash deploy_10k_subset.sh

set -e

echo "======================================"
echo "THINGS Dataset: 10k Balanced Subset"
echo "======================================"
echo ""

# Configuration
SOURCE_DIR="/mnt/VLR/data/THINGS/Images"
OUTPUT_DIR="/mnt/VLR/data/THINGS_10k_balanced"
SCRIPT_NAME="create_10k_balanced_subset.py"

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory not found: $SOURCE_DIR"
    echo "Please update SOURCE_DIR in this script or ensure THINGS dataset is downloaded."
    exit 1
fi

# Count total images in source
echo "Checking source dataset..."
TOTAL_IMAGES=$(find "$SOURCE_DIR" -type f \( -name "*.jpg" -o -name "*.png" \) | wc -l)
TOTAL_CONCEPTS=$(find "$SOURCE_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)

echo "Source dataset statistics:"
echo "  Total images: $TOTAL_IMAGES"
echo "  Total concepts: $TOTAL_CONCEPTS"
echo "  Expected images per concept: ~$((10000 / TOTAL_CONCEPTS))"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Install requirements if needed
echo "Checking Python dependencies..."
python3 -c "import tqdm" 2>/dev/null || pip install tqdm --quiet

# Run the subset creation
echo "Creating balanced 10k subset..."
echo "This may take a few minutes..."
echo ""

python3 "$SCRIPT_NAME" \
    --source "$SOURCE_DIR" \
    --output "$OUTPUT_DIR" \
    --count 10000 \
    --seed 42

# Verify the subset
echo ""
echo "Verifying subset..."
SUBSET_COUNT=$(find "$OUTPUT_DIR" -type f \( -name "*.jpg" -o -name "*.png" \) | wc -l)
echo "  Images in subset: $SUBSET_COUNT"

# Check disk usage
SUBSET_SIZE=$(du -sh "$OUTPUT_DIR" | cut -f1)
echo "  Disk usage: $SUBSET_SIZE"

# Backup to S3 if AWS CLI is available
if command -v aws &> /dev/null; then
    echo ""
    read -p "Upload subset to S3? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        S3_BUCKET="s3://vlr-antara-data/THINGS_10k_balanced/"
        echo "Uploading to $S3_BUCKET..."
        aws s3 sync "$OUTPUT_DIR" "$S3_BUCKET" --profile VLR_project
        echo "✓ Upload complete!"
    fi
fi

echo ""
echo "======================================"
echo "Subset creation complete!"
echo "======================================"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "1. Review subset_metadata.json for statistics"
echo "2. Update your StyleGAN2 config to use: $OUTPUT_DIR"
echo "3. Start training with 25 epochs"
echo ""
