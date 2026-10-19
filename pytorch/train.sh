#!/usr/bin/env bash
# Stream the autoplay log through the producer into the trainer.
#
#   ./train.sh                                            # CNN, best.pt
#   ./train.sh --arch transformer --ckpt best-tf.pt --csv loss_tf.csv
#
# Any arguments are passed to training.py. DATA overrides the input file.
set -o pipefail        # makes the shell return the *first* non-zero exit

DATA="${DATA:-$HOME/data/autoplay-softmax-v-hasty-5.txt}"

# Activate the virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "venv/bin/activate not found!"
    exit 1
fi

cat "$DATA" | \
  ( ../bin/mlproducer ; echo "producer exit=$?" >&2 ) | \
  ( pv -br ;            echo "pv exit=$?"         >&2 ) | \
  ( python training.py "$@" ; echo "training exit=$?" >&2 )
echo "pipeline exit=$?"
