# A binary builder for Macondo

from invoke import task


@task
def build(c):
    c.run("go build")
    c.run("go build ./cmd/make_gaddag")
    c.run(
        "tar -czvf macondo-darwin.tar.gz --exclude='./data/lexica/gaddag/NWL18.gaddag' ./macondo ./make_gaddag ./data"
    )
