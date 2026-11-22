#!/bin/bash
# usage: common/run.sh LOGFILE cmd args...   -- logs command, start/end time, duration; tees output.
# Guard: runs only when env NCM_GEN equals the content of common/GEN (stale queues skip their jobs).
# Records the PID of the started job (MSYS pid and Windows pid) in common/pids.log at start time.
D="$(dirname "$0")"; G=$(cat "$D/GEN" 2>/dev/null)
log=$1; shift
if [ "$NCM_GEN" != "$G" ]; then echo "### SKIPPED (stale queue, no token): $*" >> "$D/skipped.log"; exit 0; fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
{ echo "### CMD: $*"; echo "### START: $(date -Iseconds)"; } | tee -a "$log"
s=$(date +%s)
"$@" > >(tee -a "$log") 2>&1 &
pid=$!
echo "$pid winpid=$(cat /proc/$pid/winpid 2>/dev/null) $(date -Iseconds) $*" >> "$D/pids.log"
wait $pid; rc=$?
sleep 1
e=$(date +%s)
echo "### END: $(date -Iseconds) rc=$rc duration_s=$((e-s))" | tee -a "$log"
exit $rc
