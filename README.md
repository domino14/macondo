# macondo

A crossword board game solver. It may be the best one in the world (so far).

Current master build status:
[![domino14](https://circleci.com/gh/domino14/macondo.svg?style=svg)](https://circleci.com/gh/domino14/macondo)

# How to use Macondo:

See the manual and information here:

https://domino14.github.io/macondo

# protoc

To generate pb files, run this in the macondo directory:

`protoc --go_out=gen --go_opt=paths=source_relative ./api/proto/macondo/macondo.proto`

Make sure you have done

`go install google.golang.org/protobuf/cmd/protoc-gen-go@latest`

### Attributions

Wolges-awsm is Copyright (C) 2020-2022 Andy Kurnia and released under the MIT license. It can be found at https://github.com/andy-k/wolges-awsm/. Macondo interfaces with it as a server.

KLV and KWG are Andy Kurnia's leave and word graph formats. They are small and fast! See more info at https://github.com/andy-k/wolges

Some of the code for the endgame solver was influenced by the MIT-licensed Chess solver Blunder. See code at https://github.com/algerbrex/blunder