#!/usr/bin/env bash
# Health check for running Modal containers: GPU util / VRAM, container RAM, run progress.
# Usage: scripts/modal_health.sh   (from the repo root; reads logs/*.log for run progress)
set -u
ids=$(modal container list --json 2>/dev/null | python -c "import json,sys; [print(c['container_id'], c['app_id'], c['start_time']) for c in json.load(sys.stdin)]")
[ -z "$ids" ] && { echo "no running containers"; exit 0; }
while read -r cid app start; do
  echo "=== $cid ($app, started $start)"
  timeout 60 modal container exec "$cid" -- bash -c '
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader
    c=/sys/fs/cgroup; if [ -f $c/memory.current ]; then echo "container RAM: $(( $(cat $c/memory.current)/1073741824 )) GiB used (limit: $(cat $c/memory.max))"; else awk "/MemTotal|MemAvailable/{printf \"%s %d GiB  \", \$1, \$2/1048576} END{print \"\"}" /proc/meminfo; fi
    df -h /tmp | awk "NR==2{print \"/tmp disk used: \"\$3\"/\"\$2}"' 2>&1 | grep -v "^$"
done <<< "$ids"
echo "=== run progress (last [stew] line per log)"
for f in logs/*.log; do [ -f "$f" ] && printf "%-28s %s\n" "$(basename $f)" "$(grep -a '\[stew\]' "$f" | tail -1 | cut -c1-110)"; done
