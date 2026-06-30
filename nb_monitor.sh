#!/usr/bin/env bash
# nb_monitor.sh — poll a running nbconvert log every 60s and print a one-liner
# Usage: bash nb_monitor.sh <log_path> [pid]
LOG="$1"
PID="$2"

if [[ -z "$LOG" ]]; then
  echo "Usage: nb_monitor.sh <log_path> [pid]"
  exit 1
fi

while true; do
  TS=$(date '+%H:%M:%S')
  # Check if nbconvert is still running
  if [[ -n "$PID" ]]; then
    RUNNING=$(ps -p "$PID" --no-headers 2>/dev/null | wc -l)
  else
    RUNNING=$(ps aux | grep nbconvert | grep -v grep | wc -l)
  fi

  # Latest artifact written (newest file in working dir)
  LATEST=$(ls -1t /kaggle/working/$(dirname "$LOG" | xargs basename 2>/dev/null)/ 2>/dev/null | head -1)

  # Last non-empty log line
  LAST=$(grep -v '^$' "$LOG" 2>/dev/null | tail -1)

  echo "[$TS] running=$RUNNING  latest=$LATEST  log: $LAST"

  if [[ "$RUNNING" -eq 0 ]]; then
    echo "[$TS] nbconvert finished."
    break
  fi
  sleep 60
done
