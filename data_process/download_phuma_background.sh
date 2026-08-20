#!/bin/bash
nohup .venv_sim/bin/python download_phuma.py > /tmp/phuma_download.log 2>&1 &
echo "Started download in background"
