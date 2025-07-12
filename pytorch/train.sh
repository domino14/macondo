#!/usr/bin/env bash
set -o pipefail        # makes the shell return the *first* non-zero exit

# Activate the virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "venv/bin/activate not found!"
    exit 1
fi

cat ~/data/autoplay-softmax-v-hasty-5.txt | \
  ( ../bin/mlproducer ; echo "producer exit=$?" >&2 ) | \
  ( pv -br ;            echo "pv exit=$?"         >&2 ) | \
  ( python training.py ;        echo "training exit=$?"   >&2 )
echo "pipeline exit=$?"