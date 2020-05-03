# A binary builder for Macondo

from invoke import task


@task
def build(c):
    c.run("go build")
    c.run("go build ./cmd/make_gaddag")
    c.run(
        "tar -czvf macondo-darwin.tar.gz "
        "./macondo ./make_gaddag ./data ./shell/helptext"
    )
