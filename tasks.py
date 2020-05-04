# A binary builder for Macondo

from invoke import task


@task
def build(c):
    c.run("go build")
    c.run("go build ./cmd/make_gaddag")
    c.run("go build ./cmd/make_leaves_structure")
    c.run(
        "tar -czvf macondo-darwin.tar.gz "
        "./macondo ./make_gaddag ./make_leaves_structure ./data "
        "./shell/helptext"
    )
