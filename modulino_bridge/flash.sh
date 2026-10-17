#!/usr/bin/env bash
# Flash the Modulino bridge sketch to the STM32H5F5 without /dev/ttyACM0 contention.
# The arduino-router service owns the serial port, so it is stopped for the upload and restarted after.
sudo systemctl disable --now arduino-router.service && arduino-cli compile --fqbn arduino:zephyr:ventunoq ~/modulino_bridge && arduino-cli upload -p /dev/ttyACM0 --fqbn arduino:zephyr:ventunoq ~/modulino_bridge && sudo systemctl enable --now arduino-router.service
