# A binary builder for Macondo

from invoke import task


@task
def build(c):
    tag = c.run("git describe --exact-match --tags", hide=True).stdout.strip()
    print("Tag was", tag)
    c.run("go build -o macondo ./cmd/shell")
    c.run("go build ./cmd/make_gaddag")
    c.run("go build ./cmd/make_leaves_structure")
    c.run(
        "tar --exclude './data/strategy/default_english/*.idx' "
        "--exclude './data/letterdistributions' "
        f"-czvf macondo-darwin-{tag}.tar.gz "
        "./macondo ./make_gaddag ./make_leaves_structure ./data "
        "./shell/helptext"
    )
