# Boltzmann

[![test status](https://github.com/djbarker/boltzmann/actions/workflows/python-app.yml/badge.svg)](https://github.com/djbarker/boltzmann/actions/workflows/python-app.yml)
[![docs](https://img.shields.io/badge/documentation-782bc8)](https://djbarker.github.io/boltzmann/)
[![gallery](https://img.shields.io/badge/gallery-1a9988)](gallery)
[![examples](https://img.shields.io/badge/examples-9a9400)](python/examples)

A GPU accelerated Python package for running [Lattice-Boltzmann](https://en.wikipedia.org/wiki/Lattice_Boltzmann_methods) simulations.

<p align="center">
    <picture align="center">
        <a href="https://github.com/djbarker/boltzmann/tree/master/gallery">
            <img src="gallery/images/banner.png"/>
        </a>
    </picture>
    </br>
    <i>
    2D flow around a cylinder with Re=10,000 and 27 million grid cells.
    </i>
</p>

## Highlights 

- 📚 [Fully documented.](https://djbarker.github.io/boltzmann/)
- 🔢 Simple interface; simulation data is exposed to Python as [`numpy`](https://numpy.org/) arrays.
- 🚀 Accelerated with [OpenCL](https://en.wikipedia.org/wiki/OpenCL).
- 📦 Support for 2D and 3D simulations.
- 💾 Save & re-load your simulations (checkpointing).
- 📏 Utilities to map from physical to simulation units.
- 🎨 [Advection-diffusion](https://en.wikipedia.org/wiki/Convection%E2%80%93diffusion_equation) kernels for tracers.
- 🔥 Coupling of tracers to fluid via the [Boussinesq approximation](https://en.wikipedia.org/wiki/Boussinesq_approximation_(buoyancy)).
- 💥 Choice of BGK or TRT collision operators.
- 🔎 Utilities to output `.vti` files for analysis in [ParaView](https://www.paraview.org/)

## Get Started

> [!NOTE]
> I have run this module on Linux only (Ubuntu 24 natively, and Ubuntu 22 via WSL 2).
> There is nothing explicitly Linux specific in the code but consider this a heads-up.

### Installation

Currently the installation of this package is very manual.
To install from source you will need something to manage your Python environments ([`micromamba`](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html), for example) and a working installation of [Rust](https://www.rust-lang.org/tools/install). To actually run any simulations you will need a working OpenCL setup.

The following steps will set-up a Python environment and (editable) install the `boltzmann` module:

```bash
# create & use a new environment
$ micromamba create -n boltzmann python=3.10 maturin && micromamba activate boltzmann  
# download the repo
$ git clone git@github.com:djbarker/boltzmann.git && cd boltzmann
# build & install the module
$ maturin develop --release
```

### Running

If the steps above all worked you should now be able to use the module. 
Setting up a minimal simulation is very simple.
This example demonstrates running a small 2D simulation on the CPU and showing the result with [`matplotlib`](https://matplotlib.org/).

https://github.com/djbarker/boltzmann/blob/999d8b2557ab94f2ed5151ec678b43af87a8565d/python/examples/basic.py#L1-L27

This should produce the plot below

<p align="center">
    <picture align="center">
        <img src="gallery/images/example_basic.png"/>
    </picture>
    </br>
    <i>
    Basic 2D simulation showing <a href="https://en.wikipedia.org/wiki/Kelvin%E2%80%93Helmholtz_instability">Kelvin-Helmoltz instability</a>.
    </i>
</p>

The code for this example is at [`examples/basic.py`](/python/examples/basic.py).

For further explanation and more functionality check out [the full docs](https://djbarker.github.io/boltzmann/).

## Examples

There are some examples of simulations run with this package in the [gallery](/gallery/), the code for which is available under the [`examples`](/python/examples/) directory.
