#!/bin/sh

# Download raw data from Zenodo into data/raw/

mkdir -p data/raw

URL_MISSION1="https://zenodo.org/records/15237121/files/ESA-Mission1.zip?download=1"
URL_MISSION2="https://zenodo.org/records/15237121/files/ESA-Mission2.zip?download=1"

MISSION="${1:-all}"

if [ "$MISSION" = "1" ] || [ "$MISSION" = "all" ]; then
  echo "Downloading ESA-Mission1.zip..."
  curl -L "$URL_MISSION1" -o data/raw/ESA-Mission1.zip
fi

if [ "$MISSION" = "2" ] || [ "$MISSION" = "all" ]; then
  echo "Downloading ESA-Mission2.zip..."
  curl -L "$URL_MISSION2" -o data/raw/ESA-Mission2.zip
fi

echo "Download completed. Files are saved in data/raw/."
