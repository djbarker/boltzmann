#!/usr/bin/env bash

set -e

if [ -z "$(git status --porcelain)" ]; then
    echo "Working directory is clean."
    echo "Deploying."
else
    echo "Uncommited changes or untracked files in the working directory."
    echo "Aborting deploy."
    exit 1
fi

make build

cp -r build/html/* ../

git add -u
git commit -m "Update docs."
git push