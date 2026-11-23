#!/bin/sh
# Re-download the InferenceX server logs behind server_log_memory.json (one benchmark result per configuration, ids in
# log_ids.txt; ~260 MB) and re-parse them:  sh fetch_logs.sh <out-dir>  then  node parse_logs.js <out-dir> server_log_memory.json
out=${1:-logs}; mkdir -p "$out"
while read -r id; do
  [ -s "$out/$id.log" ] || curl -sS -m 180 --retry 3 "https://inferencex.semianalysis.com/api/v1/server-log?id=$id&download=1" -o "$out/$id.log"
done < "$(dirname "$0")/log_ids.txt"
