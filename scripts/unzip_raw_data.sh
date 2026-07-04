#!/bin/sh

# Unzip all .zip files in data/raw/ and delete them after extraction

if [ ! -d data/raw ]; then
  echo "data/raw directory does not exist."
  exit 1
fi

cd data/raw

for zipfile in *.zip; do
  if [ -e "$zipfile" ]; then
    echo "Unzipping $zipfile..."
    unzip -o "$zipfile"
    if [ $? -eq 0 ]; then
      echo "Deleting $zipfile..."
      rm "$zipfile"
    else
      echo "Failed to unzip $zipfile."
    fi
  fi
done

echo "All zip files processed."
