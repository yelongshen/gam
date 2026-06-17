#!/bin/bash
#
# Full SONIC Data Processing Pipeline
# Monitors tar extraction and runs data processing
#

set -e

DATASET_ROOT="/home/grease/ego_dataset/work_bearlu/data/bones-studio-seed"
OUTPUT_DIR="/home/grease/ego_dataset/work_bearlu/data/bones-studio-processed"
CONFIG_PATH="/home/grease/gam/gear_sonic/training/config_sonic_data.yaml"
VENV_PATH="/home/grease/gam/.venv_sim"

echo "════════════════════════════════════════════════════════════════"
echo "SONIC Full Data Processing Pipeline"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Dataset root: $DATASET_ROOT"
echo "Output dir:   $OUTPUT_DIR"
echo "Config:       $CONFIG_PATH"
echo ""

# Check tar extraction status
check_extraction() {
    local g1_done=0
    local soma_p_done=0
    local soma_u_done=0
    
    [[ -d "$DATASET_ROOT/g1" ]] && g1_done=1
    [[ -d "$DATASET_ROOT/soma_proportional" ]] && soma_p_done=1
    [[ -d "$DATASET_ROOT/soma_uniform" ]] && soma_u_done=1
    
    echo "✓ G1 extracted: $([ $g1_done -eq 1 ] && echo 'YES' || echo 'NO')"
    echo "✓ SOMA proportional extracted: $([ $soma_p_done -eq 1 ] && echo 'YES' || echo 'NO')"
    echo "✓ SOMA uniform extracted: $([ $soma_u_done -eq 1 ] && echo 'YES' || echo 'NO')"
    
    if [[ $g1_done -eq 1 && $soma_p_done -eq 1 ]]; then
        return 0  # Ready to process
    else
        return 1  # Not ready
    fi
}

# Monitor extraction
echo "Monitoring tar extraction..."
echo ""

while ! check_extraction; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Waiting for tar extraction to complete..."
    sleep 30
done

echo ""
echo "✓ All required files extracted!"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Starting data processing..."
echo "════════════════════════════════════════════════════════════════"
echo ""

# Activate venv and run processing
source "$VENV_PATH/bin/activate"

python gear_sonic/training/process_sonic_data.py --config "$CONFIG_PATH"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✓ Data processing complete!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Output: $OUTPUT_DIR"
echo ""
ls -lh "$OUTPUT_DIR" | head -20

