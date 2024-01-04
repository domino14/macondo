#!/bin/bash

set -euo pipefail

for lex in "NWL20" "NWL18" "America" "CSW21" "CSW19" "ECWL" "FRA20"
do
    awk '{print $1}' "$DATA_PATH/lexica/$lex.txt" > "$DATA_PATH/lexica/$lex-stripped.txt"
    awk '{print toupper($0)}' "$DATA_PATH/lexica/$lex-stripped.txt" > "$DATA_PATH/lexica/$lex-toupper.txt"
    echo "lex $lex"

    CONTAINER_ID="$(docker create kbuilder -- english-kwg /home/in.txt /home/out.kwg )"
    trap "docker rm $CONTAINER_ID" EXIT
    echo "$CONTAINER_ID"

    docker cp "$DATA_PATH/lexica/$lex-toupper.txt" "$CONTAINER_ID:/home/in.txt"
    docker start "$CONTAINER_ID"
    docker attach "$CONTAINER_ID" || true
    docker cp "$CONTAINER_ID:/home/out.kwg" "$DATA_PATH/lexica/gaddag/$lex.kwg"

    docker rm "$CONTAINER_ID"
    trap "" EXIT

    echo "after $lex"
done

for lex in "OSPS44"
do
    awk '{print $1}' "$DATA_PATH/lexica/$lex.txt" > "$DATA_PATH/lexica/$lex-stripped.txt"
    echo "lex $lex"

    CONTAINER_ID="$(docker create kbuilder -- polish-kwg /home/in.txt /home/out.kwg )"
    trap "docker rm $CONTAINER_ID" EXIT
    echo "$CONTAINER_ID"

    docker cp "$DATA_PATH/lexica/$lex-stripped.txt" "$CONTAINER_ID:/home/in.txt"
    docker start "$CONTAINER_ID"
    docker attach "$CONTAINER_ID" || true
    docker cp "$CONTAINER_ID:/home/out.kwg" "$DATA_PATH/lexica/gaddag/$lex.kwg"

    docker rm "$CONTAINER_ID"
    trap "" EXIT

    echo "after $lex"
done


for lex in "NSF22"
do
    awk '{print $1}' "$DATA_PATH/lexica/$lex.txt" > "$DATA_PATH/lexica/$lex-stripped.txt"
    awk '{print toupper($0)}' "$DATA_PATH/lexica/$lex-stripped.txt" > "$DATA_PATH/lexica/$lex-toupper.txt"
    echo "lex $lex"

    CONTAINER_ID="$(docker create kbuilder -- norwegian-kwg /home/in.txt /home/out.kwg )"
    trap "docker rm $CONTAINER_ID" EXIT
    echo "$CONTAINER_ID"

    docker cp "$DATA_PATH/lexica/$lex-toupper.txt" "$CONTAINER_ID:/home/in.txt"
    docker start "$CONTAINER_ID"
    docker attach "$CONTAINER_ID" || true
    docker cp "$CONTAINER_ID:/home/out.kwg" "$DATA_PATH/lexica/gaddag/$lex.kwg"

    docker rm "$CONTAINER_ID"
    trap "" EXIT

    echo "after $lex"
done

echo "done creating kwgs"
