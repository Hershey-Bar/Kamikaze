#!/usr/bin/env bash
# Start the Kamikaze receiver on the VENTUNO Q (port 5000). Extra args are passed to receiver.py.
cd "$(dirname "$0")"
exec .venv/bin/python receiver.py "$@"
