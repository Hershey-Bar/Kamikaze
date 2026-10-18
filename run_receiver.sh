#!/usr/bin/env bash
# Start the Kamikaze receiver on the VENTUNO Q (port 5000). Extra args are passed to receiver.py.
cd "$(dirname "$0")"
# When stdout is a log file, keep it under KAMIKAZE_LOG_MAX_MB so a noisy native runtime can never fill the eMMC.
LOG_MAX_MB=${KAMIKAZE_LOG_MAX_MB:-200}
LOG=$(readlink -f /proc/$$/fd/1 2>/dev/null)
if [ -f "$LOG" ]; then
  ( pid=$$; while kill -0 "$pid" 2>/dev/null; do
      sleep 30
      [ "$(stat -c %s "$LOG" 2>/dev/null || echo 0)" -gt $((LOG_MAX_MB * 1048576)) ] && truncate -s 0 "$LOG"
    done ) &
fi
exec .venv/bin/python receiver.py "$@"
