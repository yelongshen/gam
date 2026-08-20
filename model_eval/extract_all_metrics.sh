#!/bin/bash
set -e
echo "Extracting all AMASS archives..."
mkdir -p ~/egodata/downloads/amass/extracted

# Extract LAFAN1 if not already
unzip -q -o ~/egodata/downloads/lafan1/lafan1.zip -d ~/egodata/downloads/lafan1_extracted/ || true

# Extract AMASS
cd ~/egodata/downloads/amass
for arc in *.tar.bz2; do
    folder_name="${arc%.tar.bz2}"
    if [ ! -d "extracted/$folder_name" ]; then
        echo "Extracting $arc..."
        mkdir -p "extracted/$folder_name"
        tar -xjf "$arc" -C "extracted/$folder_name"
    else
        echo "Already extracted $arc"
    fi
done
echo "Extraction Complete!"
