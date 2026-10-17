#!/bin/bash
# Restart the receiver in the background and log to /tmp/receiver.log.
for p in $(pgrep -f "python receiver.py"); do kill "$p"; done
sleep 2
cd ~/hackathon_iot
nohup ./run_receiver.sh > /tmp/receiver.log 2>&1 &
sleep 1
echo started
