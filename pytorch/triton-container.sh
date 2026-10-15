#!/usr/bin/env bash
# (Re)create the local Triton container: explicit model control, the best
# model loaded at start, host ports 8100-8102 (8000-8002 are taken by the
# liwords docker stack here; macondo's default elsewhere is 8001).
#   MODEL (macondo-nn-tf-streamopen), IMAGE (tritonserver:26.08-py3)
# Stops whatever the container is serving: not during a match or generation.
set -eu
MODEL=${MODEL:-macondo-nn-tf-streamopen}
IMAGE=${IMAGE:-nvcr.io/nvidia/tritonserver:26.08-py3}
docker rm -f macondo-triton >/dev/null 2>&1 || true
docker run -d --name macondo-triton --restart unless-stopped --gpus all \
  -p 8100:8000 -p 8101:8001 -p 8102:8002 \
  -v "$HOME/code/macondo/data/strategy/default/models:/models" \
  "$IMAGE" tritonserver --model-repository=/models --model-control-mode=explicit --load-model="$MODEL"
for i in $(seq 1 60); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8100/v2/models/$MODEL/ready)" = 200 ] && { echo "$MODEL ready"; exit 0; }
  sleep 2
done
echo "$MODEL not ready after 2 min"; docker logs macondo-triton 2>&1 | tail -20; exit 1
