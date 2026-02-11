#!/bin/bash
# Download fraud detection dataset from Kaggle

# Create download directory if it doesn't exist
mkdir -p ~/Downloads

# Download the dataset
echo "Downloading transactions-fraud-datasets from Kaggle..."
curl -L -o ~/Downloads/transactions-fraud-datasets.zip \
  https://www.kaggle.com/api/v1/datasets/download/computingvictor/transactions-fraud-datasets

# Check if download was successful
if [ -f ~/Downloads/transactions-fraud-datasets.zip ]; then
    echo "✓ Download successful!"
    
    # Extract the dataset
    echo "Extracting dataset..."
    unzip -q -o ~/Downloads/transactions-fraud-datasets.zip -d ~/Downloads/
    
    echo "✓ Extraction complete!"
    echo "Dataset ready at: ~/Downloads/transactions-fraud-datasets/"
else
    echo "✗ Download failed. Please check your internet connection and Kaggle credentials."
    exit 1
fi
