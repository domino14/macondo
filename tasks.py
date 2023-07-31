# A binary builder for Macondo

from invoke import task


@task
def build(c):
    tag = c.run("git describe --exact-match --tags", hide=True).stdout.strip()
    print("Tag was", tag)
    c.run("go build -o macondo ./cmd/shell")
    c.run(
        "tar --exclude './data/strategy_default_english/quackle*' "
        f"-czvf macondo-darwin-{tag}.tar.gz "
        "./macondo ./data "
    )
