set -e
for lex in "NWL20" "NWL18" "America" "CSW21" "CSW19"
do
    awk '{print $1}' $LEXICON_PATH/$lex.txt > $LEXICON_PATH/$lex-stripped.txt
    docker run --rm -v $LEXICON_PATH:/opt kbuilder -- english-kwg /opt/$lex-stripped.txt /opt/gaddag/$lex.kwg
done

awk '{print $1}' $LEXICON_PATH/OSPS44.txt > $LEXICON_PATH/OSPS44-stripped.txt
docker run --rm -v $LEXICON_PATH:/opt kbuilder -- polish-kwg /opt/OSPS44-stripped.txt /opt/gaddag/OSPS44.kwg

echo "done creating kwgs"