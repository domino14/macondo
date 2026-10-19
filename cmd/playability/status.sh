#!/bin/bash
# Progress of a playability gen run.
D=${PLAY_DIR:-$HOME/data/playability}
echo "gen:   $(ls $D/gen/shard-*.gob.gz 2>/dev/null | wc -l)/1000 shards; pid $(cat $D/gen.pid) $(ps -p $(cat $D/gen.pid) >/dev/null && echo running || echo stopped)"
tail -1 $D/gen.log | sed 's/\x1b\[[0-9;]*m//g'
uptime
