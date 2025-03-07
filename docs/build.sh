#!/usr/bin/bash

make clean

# Note: the paths following ../python/boltzmann are excludes.
sphinx-apidoc -ef \
    -o source \
    --templatedir=source/_templates \
    ../python/boltzmann \
    ../python/boltzmann/kernel_gen.py \
    ../python/boltzmann/boltzmann.*so

make html