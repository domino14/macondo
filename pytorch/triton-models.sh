#!/usr/bin/env bash
# Triton model control (explicit mode, localhost:8100).
#   triton-models.sh list            names of the READY models
#   triton-models.sh unload-all      unload every READY model (frees its GPU memory)
#   triton-models.sh unload NAME
#   triton-models.sh load NAME [VER] load and wait until ready
# A training run must not share the card with served models: the desktop
# already takes 2-2.5 GB of the 8 GB, so the drivers unload everything before
# training and load only the model a match needs, then unload it after.
set -u
URL=${TRITON_HTTP:-localhost:8100}
ready() { curl -s -X POST "$URL/v2/repository/index" | python3 -c "import sys,json; [print(m['name']) for m in json.load(sys.stdin) if m.get('state')=='READY']" 2>/dev/null; }
case "${1:-}" in
  list) ready ;;
  unload-all)
    for m in $(ready); do
      printf '%s unload http %s\n' "$m" "$(curl -s -o /dev/null -w '%{http_code}' -X POST "$URL/v2/repository/models/$m/unload")"
    done ;;
  unload)
    printf '%s unload http %s\n' "$2" "$(curl -s -o /dev/null -w '%{http_code}' -X POST "$URL/v2/repository/models/$2/unload")" ;;
  load)
    name=$2; ver=${3:-1}
    printf '%s load http %s\n' "$name" "$(curl -s -o /dev/null -w '%{http_code}' -X POST "$URL/v2/repository/models/$name/load")"
    for i in $(seq 1 30); do
      [ "$(curl -s -o /dev/null -w '%{http_code}' "$URL/v2/models/$name/versions/$ver/ready")" = 200 ] && { echo "$name v$ver ready"; exit 0; }
      sleep 5
    done
    echo "$name v$ver NOT ready"; exit 1 ;;
  *) echo "usage: $0 list|unload-all|unload NAME|load NAME [VER]"; exit 2 ;;
esac
