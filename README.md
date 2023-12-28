# MaxwellFDTD

[![Build Status](https://github.com/gioelemo/MaxwellFDTD.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/gioelemo/MaxwellFDTD.jl/actions/workflows/CI.yml/badge.svg?branch=main)

## Contents

- [Introduction](#introduction)
- [Setup](#setup)
- [Mathematical formulation](#mathematical-formulation)
- [Numerical Methods](#numerical-methods)
- [1D FDTD](#1d-fdtd)
- [2D FDTD](#2d-fdtd)
- [3D FDTD](#3d-fdtd)
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

Faraday's law
$$\nabla \times \boldsymbol{E} = - \mu\frac{\partial\boldsymbol{H}}{\partial t} \tag{1} $$

Ampere's law
$$\nabla \times \boldsymbol{H} = J_c + \epsilon\frac{\partial\boldsymbol{E}}{\partial t} \tag{2}$$

## Numerical Methods

TODO: Explain finite difference (and the method itself)

To solve the equations it is possible to use the Finite Difference Time Domain Method (FDTD).

This method introduced by Kane S. Yee [1] consists of discretizing the time-dependent Maxwell's equation using a central finite-difference approach.

The finite-difference equations derived from this process are addressed in a leapfrog fashion, either through software or hardware. Initially, the electric field vector components within a specific spatial volume are resolved at a particular moment in time. Subsequently, the magnetic field vector components in the same spatial domain are addressed in the subsequent time step. This iterative process continues until the anticipated transient or steady-state electromagnetic field behavior is completely developed [2].

## 1D FDTD
The goal of this section is to provide a simple code for a Finite Difference Time domain simulator for solving a simple version of the Maxwell equations in 1D.

We assume in this case that the electric field only has a $z$ component.

In this case Faraday's law (Equation 1) can be written as:

$$
-\mu \frac{\partial \boldsymbol{H}}{\partial t}=\nabla \times \boldsymbol{E}=\left|\begin{array}{ccc}
\hat{\boldsymbol{a}}_x & \hat{\boldsymbol{a}}_y & \hat{\boldsymbol{a}}_z \\
\frac{\partial}{\partial x} & 0 & 0 \\
0 & 0 & E_z
\end{array}\right|=-\hat{\boldsymbol{a}}_y \frac{\partial E_z}{\partial x} \tag{3} 
$$

And similarly Ampere's law (Equation 2) can be written as:

$$
\epsilon \frac{\partial \boldsymbol{E}}{\partial t}=\nabla \times \boldsymbol{H}=\left|\begin{array}{ccc}
\hat{\boldsymbol{a}}_x & \hat{\boldsymbol{a}}_y & \hat{\boldsymbol{a}}_z \\
\frac{\partial}{\partial x} & 0 & 0 \\
0 & H_y & 0
\end{array}\right|=\hat{\boldsymbol{a}}_z \frac{\partial H_y}{\partial x} \tag{4}
$$

The scalar equations form (3) and (4) in 1D are given as:

$$
\begin{align*}
\mu \frac{\partial H_y}{\partial t} &= \frac{\partial E_z}{\partial x} \tag{5}\\
\epsilon \frac{\partial E_z}{\partial t} &= \frac{\partial H_y}{\partial x} \tag{6}
\end{align*}
$$
We can then transform the previous two equation using a finite difference approach as follow:

1. For $H_y$:
$$\begin{align*}
\mu \frac{H_y^{q+\frac{1}{2}}\left[m+\frac{1}{2}\right]-H_y^{q-\frac{1}{2}}\left[m+\frac{1}{2}\right]}{\Delta_t}&=\frac{E_z^q[m+1]-E_z^q[m]}{\Delta_x} \\

H_y^{q+\frac{1}{2}}\left[m+\frac{1}{2}\right]&=H_y^{q-\frac{1}{2}}\left[m+\frac{1}{2}\right]+\frac{\Delta_t}{\mu \Delta_x}\left(E_z^q[m+1]-E_z^q[m]\right)
\end{align*}
$$

2.  For $E_z$:
$$
\begin{align*}
\epsilon \frac{E_z^{q+1}[m]-E_z^q[m]}{\Delta_t}&=\frac{H_y^{q+\frac{1}{2}}\left[m+\frac{1}{2}\right]-H_y^{q+\frac{1}{2}}\left[m-\frac{1}{2}\right]}{\Delta_x}\\
E_z^{q+1}[m]&=E_z^q[m]+\frac{\Delta_t}{\epsilon \Delta_x}\left(H_y^{q+\frac{1}{2}}\left[m+\frac{1}{2}\right]-H_y^{q+\frac{1}{2}}\left[m-\frac{1}{2}\right]\right)
\end{align*}
$$

where:
- $E_z^{q+1}[m]$: Electric field component $E_z$ at spatial position $m$ and time step $q+1$.

- $E_z^q[m]$: Electric field component $E_z$ at spatial position $m$ and time step $q$.

- $\epsilon$: Permittivity of the medium.

- $\mu$: Permeability of the medium.

- $\Delta_t$: Time step size.

- $\Delta_x$: Spatial step size.

- $H_y^{q+\frac{1}{2}}\left[m+\frac{1}{2}\right]$: Magnetic field component $H_y$ at the half-integer spatial position $m+\frac{1}{2}$ and time step $q+\frac{1}{2}$.

- $H_y^{q+\frac{1}{2}}\left[m-\frac{1}{2}\right]$: Magnetic field component $H_y$ at the half-integer spatial position $m-\frac{1}{2}$ and time step $q+\frac{1}{2}$.

We define the Courant number $S_c$ as
$$S_c := \frac{c \Delta_t}{\Delta_x}$$

We can use the following update equations when working with integer indexes and assuming that $S_c=1$:

- The magnetic-field nodes can be updated with:
`hy[m] = hy[m] + (ez[m + 1] - ez[m]) / imp0`

- The electric-field nodes can be updated with: `ez[m] = ez[m] + (hy[m] - hy[m - 1]) * imp0`

where `imp0` is the characteristic impedance of free space (approximately 377 $\Omega$).

This method is implemented in the [1D_additive_source_lossy_layer.jl](./scripts/1D_additive_source_lossy_layer.jl) file.

In this code is also additionally implemented:

1. Additive source in an explicit point of the domain (i.e. at the TSFS boundary). This source can be a Gaussian function (with specified width and location) or a sinusoidal function. This is done in the `correct_E_z()` and  `correct_E_z2()` functions. 

2. A Total-Field/Scattered-Field (TFSF) Boundary which separate the total field into incident and scattered components, allowing for accurate characterization of the scattered electromagnetic waves in the vicinity of the simulation domain.

3. An absorbing boundary condition (ABC) on the left part of the $E_z$ field to simulate an open and infinite environment by introducing a boundary that absorbs outgoing waves, minimizing reflections from the simulation domain boundaries. This is done in `ABC_bc()` function.

4. An interface index between free space and dielectric space (controlled by the `interface_index` parameter) 

5. A lossy region where some loss is introduced (controlled by the `loss_layer_index` , `epsR` and `loss` variables).


## 2D FDTD

TODO: Explain formulas in 2D + results

The update equations in 1D are given as:


### $TM^z$
$$
\begin{align*}
-\sigma_m H_x - \mu \frac{\partial H_x}{\partial t} &= \frac{\partial E_z}{\partial y} \\
\sigma_m H_y + \mu \frac{\partial H_y}{\partial t} &= \frac{\partial E_z}{\partial x} \\
\sigma E_z + \epsilon \frac{\partial E_z}{\partial t} &= \frac{\partial H_y}{\partial x} -\frac{\partial H_x}{\partial y}
\end{align*}
$$

### $TE^z$

$$
\begin{align*}
\sigma E_x + \epsilon \frac{\partial E_x}{\partial t} &= \frac{\partial H_z}{\partial y} \\
\sigma E_y + \epsilon \frac{\partial E_y}{\partial t} &= -\frac{\partial H_z}{\partial x} \\
-\sigma_m H_z - \mu\frac{\partial H_z}{\partial t} &= \frac{\partial E_y}{\partial x} - \frac{\partial E_x}{\partial y}
\end{align*}
$$

We have the following three animations:
![](./docs/Maxwell_2D_xpu_alpha=000.gif)

![](./docs/Maxwell_2D_xpu_alpha=010.gif)

![](./docs/Maxwell_2D_xpu_alpha=500.gif)

## 3D FDTD

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
