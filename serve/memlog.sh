#!/usr/bin/env bash
# Log host + container memory every 10 s: cgroup anon / file / shmem / unevictable, host MemAvailable.
NAME=${1:-dsv41-flash-exl3}
while true; do
  cid=$(docker inspect -f '{{.Id}}' "$NAME" 2>/dev/null)
  ma=$(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo)
  if [[ -n $cid && -f /sys/fs/cgroup/system.slice/docker-$cid.scope/memory.stat ]]; then
    c=/sys/fs/cgroup/system.slice/docker-$cid.scope
    cur=$(awk '{printf "%.1f", $1/1073741824}' $c/memory.current)
    st=$(awk '/^(anon|file|shmem|unevictable|file_mapped) /{printf "%s=%.1f ", $1, $2/1073741824}' $c/memory.stat)
    echo "$(date +%T) host_avail=${ma}G cg_current=${cur}G $st"
  else
    echo "$(date +%T) host_avail=${ma}G (no container)"
  fi
  sleep 10
done
