#!/usr/bin/env bash
# Flash the Modulino bridge sketch to the STM32H5F5 without serial port contention.
# The arduino-router service owns the MCU link, so it is stopped for the upload and restarted after.
# Sketch directory defaults to this script's folder; override with: flash.sh /path/to/modulino_bridge
SKETCH="${1:-$(cd "$(dirname "$0")" && pwd)}"
sudo systemctl disable --now arduino-router.service && arduino-cli compile --fqbn arduino:zephyr:ventunoq "$SKETCH" && arduino-cli upload -p /dev/ttyACM0 --fqbn arduino:zephyr:ventunoq "$SKETCH"; rc=$?
sudo systemctl enable --now arduino-router.service
exit $rc
