<!-- Title -->
<h1 align="center">
  Melvin.py
</h1>

<!-- description -->
<p align="center">
  <strong>A Python-powered GPU-accelerated framework for solving 2D computational fluid dynamics problems 💧</strong>
</p>

<!-- Information badges -->
<p align="center">
  <a href="https://www.repostatus.org/#active">
    <img alt="Repo status" src="https://www.repostatus.org/badges/latest/active.svg?style=flat-square" />
  </a>
  <a href="https://mit-license.org">
    <img alt="MIT license" src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square">
  </a>
  <!--<a href="https://github.com/jamiejquinn/Melvin.py/discussions">-->
    <!--<img alt="Ask us anything" src="https://img.shields.io/badge/Ask%20us-anything-1abc9c.svg?style=flat-square">-->
  <!--</a>-->
  <a href="https://github.com/SciML/ColPrac">
    <img alt="ColPrac: Contributor's Guide on Collaborative Practices for Community Packages" src="https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet?style=flat-square">
  </a>
</p>

<!-- Version and documentation badges -->
<p align="center">
  <a href="https://github.com/JamieJQuinn/Melvin.py/releases">
    <img alt="GitHub tag (latest SemVer pre-release)" src="https://img.shields.io/github/v/tag/jamiejquinn/melvin.py?include_prereleases&label=latest%20version&logo=github&sort=semver&style=flat-square">
  </a>
    <a href="https://github.com/JamieJQuinn/Melvin.py/actions/workflows/pytest.yml">
    <img alt="Continuous testing" src="https://github.com/jamiejquinn/melvin.py/actions/workflows/pytest.yml/badge.svg">
  </a>
  <!--<a href="https://clima.github.io/OceananigansDocumentation/stable">-->
    <!--<img alt="Stable documentation" src="https://img.shields.io/badge/documentation-stable%20release-blue?style=flat-square">-->
  <!--</a>-->
  <!--<a href="https://clima.github.io/OceananigansDocumentation/dev">-->
    <!--<img alt="Development documentation" src="https://img.shields.io/badge/documentation-in%20development-orange?style=flat-square">-->
  <!--</a>-->
</p>

Melvin.py is a user-friendly framework for building GPU-accelerated spectral simulations of 2-dimensional computational fluid dynamics problems. This is still a work-in-progress project however the following features are functional:

- **Boundary conditions**
  - doubly periodic 
  - periodic in x, Dirichlet in z
- **Spatial discretisation**
  - Fourier spectral in both directions
  - Fourier spectral in x and finite difference in z
  - Pseudo-spectral transform for handling nonlinear advection term
  - 2nd and 4th-order accurate finite-difference derivatives
- **Time stepping schemes**
  - Explicit 2nd and 4th-order Adams-Bashforth for any discretisation
  - 2nd and 4th order predictor-corrector schemes using Adams-Bashforth and Adams-Moulton schemes
  - Semi-implicit treatment of diffusion operator in fully-spectral only
- **Incompressibility**
  - Implemented via a vorticity-streamfunction formulation
- **Fully parameterised with JSON**
- WIP **Restarting from checkpoint**
- WIP **Automatic CFL-based timestep adaptation**

## Contents

* [Installation instructions](#installation-instructions)
* [Running an example](#running-an-example)
* [Getting help](#getting-help)
* [Contributing](#contributing)

## Installation instructions

You can install the latest version of Melvin using the Python package manager `pip`

```bash
pip install melvin
```

At this time, updating should be done with care, as Oceananigans is under rapid development and breaking changes to the user API occur often. But if anything does happen, please open an issue!

**Note**: Melvin is tested with Python 3.9. Older versions of Python 3 may work but YMMV.

## Running an example

The examples can be found in the `examples` folder and provide a useful starting point. Currently the examples are

- Rayleigh-Bénard convection
- Kelvin-Helmholtz instability
- Double-diffusive convection (with formation of thermohaline staircases)

Let us run the double-diffusive convection example, `ddc.py` and place output files into a data directory. The parameters, initial conditions and boundary conditions are all set within `ddc.py` file and can be run with

```bash
mkdir data
cd data
python ../examples/ddc.py
```

## Getting help

If you are interested in using Melvin.py or are trying to figure out how to use it, please feel free to ask questions and get in touch! Please bear in mind that this is still a WIP and all documentation and functionality is subject to rapid change.

## Contributing

We want your help no matter how big or small your contribution may be. It's useful even just to look over the documentation and likely find a typo or two!

If you've found a bug or have a suggestion for how Melvin.py could be improved, I encourage you to [open an issue](https://github.com/jamiejquinn/melvin.py/issues/new) or submit a pull request.

For more information, check out our [contributor's guide](https://github.com/jamiejquinn/melvin.py/blob/master/CONTRIBUTING.md).
