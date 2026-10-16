#!/bin/bash
# Progress of the playability runs.
D=${PLAY_DIR:-$HOME/data/playability}
echo "gen:   $(ls $D/gen/shard-*.gob.gz 2>/dev/null | wc -l)/1000 shards; pid $(cat $D/gen.pid) $(ps -p $(cat $D/gen.pid) >/dev/null && echo running || echo stopped)"
tail -1 $D/gen.log | sed 's/\x1b\[[0-9;]*m//g'
echo "synth: 7-8 $(ls $D/synth78 2>/dev/null | grep -c gob.gz) chunks, 9+ $(ls $D/synth9p 2>/dev/null | grep -c gob.gz) chunks; pid $(cat $D/synth.pid) $(ps -p $(cat $D/synth.pid) >/dev/null && echo running || echo stopped)"
tail -1 $D/synth.log 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g'
uptime
