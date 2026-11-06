#!/bin/bash
# usage: run_queue.sh jobsfile ; runs jobs with 2 parallel workers
cd /home/claude/pacac
cat "$1" | xargs -P 2 -L 1 bash -c 'python3 train_run.py $0 $1 > results/logs/$0_s$1.log 2>&1'
