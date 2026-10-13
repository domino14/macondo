# A binary builder for Macondo

import os

from invoke import task

EXTRAS = "./data ./scripts/lua ./explainer"


def _go_build(c, out, pkg, gitflags, goos, goarch, cc=None):
    # cgo on: the CPU inference path (ONNX Runtime) needs it, so each
    # platform is built on its own runner rather than cross-compiled.
    env = {"GOOS": goos, "GOARCH": goarch, "CGO_ENABLED": "1"}
    if cc:
        env["CC"] = cc
    c.run(f"go build -o {out} {gitflags} {pkg}", env=env)


@task(help={"target": "mac, linux or windows: the platform of the machine running the build"})
def build(c, target):
    # A tag on release builds; a commit hash on test runs.
    tag = c.run("git describe --tags --always", hide=True).stdout.strip()
    print("Tag was", tag)

    gitflags = f'-ldflags "-X main.GitVersion={tag}"'

    if target == "mac":
        # Universal mac executable. Apple's clang targets both
        # architectures, so cgo works for the one the runner is not.
        _go_build(c, "macondo-amd64", "./cmd/shell", gitflags, "darwin", "amd64", "clang -arch x86_64")
        _go_build(c, "macondo-arm64", "./cmd/shell", gitflags, "darwin", "arm64", "clang -arch arm64")
        c.run("lipo -create -output macondo macondo-amd64 macondo-arm64")
        c.run(f"zip -r macondo-{tag}-osx-universal.zip ./macondo {EXTRAS}")
        c.run("rm -f macondo macondo-amd64 macondo-arm64")
    elif target == "linux":
        _go_build(c, "macondo", "./cmd/shell", gitflags, "linux", "amd64")
        c.run(f"zip -r macondo-{tag}-linux-x86_64.zip ./macondo {EXTRAS}")
        c.run("rm -f macondo")
    elif target == "windows":
        _go_build(c, "macondo.exe", "./cmd/shell", gitflags, "windows", "amd64")
        _go_build(c, "updater.exe", "./cmd/updater", gitflags, "windows", "amd64")
        # No zip on Windows runners; 7-Zip is preinstalled.
        c.run(f"7z a -tzip macondo-{tag}-win64.zip macondo.exe updater.exe data scripts\\lua explainer")
        for f in ("macondo.exe", "updater.exe"):
            os.remove(f)
    else:
        raise ValueError(f"unknown target {target!r}: mac, linux or windows")
