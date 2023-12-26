# MaxwellFDTD

[![Build Status](https://github.com/gioelemo/MaxwellFDTD.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/gioelemo/MaxwellFDTD.jl/actions/workflows/CI.yml/badge.svg?branch=main)

## Contents

- [Introduction](#introduction)
- [Setup](#setup)
- [Mathematical formulation](#mathematical-formulation)
- [Numerical Methods](#numerical-methods)
- [1D FTDT](#1d-ftdt)
- [2D FTDT](#2d-ftdt)
- [3D FTDT](#3d-ftdt)
- [Testing](#testing)
- [Results and conclusions](#results-and-conclusions)
- [References](#references)

TODO: Control table of contents as a list

## Introduction

TODO: Describe the problem, libraries used and gpu/cpu

This repository hosts the implementation of a Maxwell equations solver using the Finite Differences Time Domain (FDTD) method in the Julia programming language, utilizing the  [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl) and [ImplicitGlobalGrid.jl](https://github.com/eth-cscs/ImplicitGlobalGrid.jl) packages. 

We start with a simple 1D implementation and we extend the code to 2D and 3D employing both GPU and CPU architectures. The implementation was further augmented to encompass multi-xPU capabilities, leveraging MPI for communication. 

All tests were conducted locally on a MacBook Pro 2017 - 2.8 GHz Intel Core i7 quad-core processor or on the Piz Daint Supercomputer (CSCS - Lugano) using one or multiple NVIDIA Tesla P100 GPUs.

## Setup

TODO: Explain how to run the code

## Mathematical formulation

TODO: Describe with simple formulas how the code is implemented

Some formulas

$\nabla \times \boldsymbol{E} = - \frac{\partial\boldsymbol{B}}{\partial t}$

$\nabla \times \boldsymbol{H} = J_c + \frac{\partial\boldsymbol{D}}{\partial t}$

$\boldsymbol{D} = \epsilon \boldsymbol{E} $

$\boldsymbol{B} = \mu \boldsymbol{H} $

$\nabla \times \boldsymbol{E} = - \mu\frac{\partial\boldsymbol{H}}{\partial t}$


$\nabla \times \boldsymbol{H} = J_c + \epsilon\frac{\partial\boldsymbol{E}}{\partial t}$

## Numerical Methods

TODO: Explain finite difference (and the method itself)

To solve the equations it is possible to use the Finite Difference Time Domain Method (FDTD).

This method introduced by Kane S. Yee [1] consists of discretizing the time-dependent Maxwell's equation using a central finite-difference approach.

The finite-difference equations derived from this process are addressed in a leapfrog fashion, either through software or hardware. Initially, the electric field vector components within a specific spatial volume are resolved at a particular moment in time. Subsequently, the magnetic field vector components in the same spatial domain are addressed in the subsequent time step. This iterative process continues until the anticipated transient or steady-state electromagnetic field behavior is completely developed [2].

## 1D FTDT

TODO: Explain formulas in 1D + results

The update equations in 1D are given as:

$$
\begin{align}
\mu \frac{\partial H_y}{\partial t} &= \frac{\partial E_z}{\partial x} \\
\epsilon \frac{\partial E_z}{\partial t} &= \frac{\partial H_y}{\partial x}
\end{align}
$$

## 2D FTDT

TODO: Explain formulas in 2D + results

The update equations in 1D are given as:


### $TM^z$
$$
\begin{align}
-\sigma_m H_x - \mu \frac{\partial H_x}{\partial t} &= \frac{\partial E_z}{\partial y} \\
\sigma_m H_y + \mu \frac{\partial H_y}{\partial t} &= \frac{\partial E_z}{\partial x} \\
\sigma E_z + \epsilon \frac{\partial E_z}{\partial t} &= \frac{\partial H_y}{\partial x} -\frac{\partial H_x}{\partial y}
\end{align}
$$

### $TE^z$

$$
\begin{align}
\sigma E_x + \epsilon \frac{\partial E_x}{\partial t} &= \frac{\partial H_z}{\partial y} \\
\sigma E_y + \epsilon \frac{\partial E_y}{\partial t} &= -\frac{\partial H_z}{\partial x} \\
-\sigma_m H_z - \mu\frac{\partial H_z}{\partial t} &= \frac{\partial E_y}{\partial x} - \frac{\partial E_x}{\partial y}
\end{align}
$$

We have the following two animations:
![](./docs/Maxwell_2D_xpu_alpha0.00.gif)

![](./docs/Maxwell_2D_xpu_alpha0.75.gif)

## 3D FTDT

TODO: Explain formulas in 3D + results

## Testing

TODO: Explain what tests are performed

## Results and conclusions

TODO: Write some conclusion and to what extent the code can be extended

## References

[1] Kane Yee (1966). "Numerical solution of initial boundary value problems involving Maxwell's equations in isotropic media". IEEE Transactions on Antennas and Propagation. 14 (3): 302–307.

[2] Finite-difference time-domain method - Wikipedia
[https://en.wikipedia.org/wiki/Finite-difference_time-domain_method](https://en.wikipedia.org/wiki/Finite-difference_time-domain_method)

[3] Understanding the Finite-Difference Time-Domain Method, John B. Schneider, [www.eecs.wsu.edu/~schneidj/ufdtd](www.eecs.wsu.edu/~schneidj/ufdtd), 2010. (also [here](./references/ufdtd.pdf))

[4] Berenger, Jean-Pierre. "A perfectly matched layer for the absorption of electromagnetic waves." Journal of computational physics 114.2 (1994): 185-200 (also [here](./references/APerfectlyMatchedLayerfortheAbsorptionofElectromagneticWaves.pdf))




TODO: Complete the list with useful references
