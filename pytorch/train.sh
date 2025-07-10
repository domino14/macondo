#!/usr/bin/env bash
set -o pipefail        # makes the shell return the *first* non-zero exit

cat ~/data/autoplay-softmax-v-hasty-5.txt | \
  ( ../bin/mlproducer ; echo "producer exit=$?" >&2 ) | \
  ( pv -br ;            echo "pv exit=$?"         >&2 ) | \
  ( python training.py ;        echo "training exit=$?"   >&2 )
echo "pipeline exit=$?"