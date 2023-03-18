#!/bin/bash

set -euo pipefail

for lex in "NWL20" "NWL18" "America" "CSW21" "CSW19"
do
    awk '{print $1}' "$LEXICON_PATH/$lex.txt" > "$LEXICON_PATH/$lex-stripped.txt"
    echo "lex $lex"

    CONTAINER_ID="$(docker create kbuilder -- english-kwg /home/in.txt /home/out.kwg )"
    trap "docker rm $CONTAINER_ID" EXIT
    echo "$CONTAINER_ID"

    docker cp "$LEXICON_PATH/$lex-stripped.txt" "$CONTAINER_ID:/home/in.txt"
    docker start "$CONTAINER_ID"
    docker attach "$CONTAINER_ID" || true
    docker cp "$CONTAINER_ID:/home/out.kwg" "$LEXICON_PATH/gaddag/$lex.kwg"

    docker rm "$CONTAINER_ID"
    trap "" EXIT

    echo "after $lex"
done

for lex in "OSPS44"
do
    awk '{print $1}' "$LEXICON_PATH/$lex.txt" > "$LEXICON_PATH/$lex-stripped.txt"
    echo "lex $lex"

    CONTAINER_ID="$(docker create kbuilder -- polish-kwg /home/in.txt /home/out.kwg )"
    trap "docker rm $CONTAINER_ID" EXIT
    echo "$CONTAINER_ID"

    docker cp "$LEXICON_PATH/$lex-stripped.txt" "$CONTAINER_ID:/home/in.txt"
    docker start "$CONTAINER_ID"
    docker attach "$CONTAINER_ID" || true
    docker cp "$CONTAINER_ID:/home/out.kwg" "$LEXICON_PATH/gaddag/$lex.kwg"

    docker rm "$CONTAINER_ID"
    trap "" EXIT

    echo "after $lex"
done

echo "done creating kwgs"
