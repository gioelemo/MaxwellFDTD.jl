# MaxwellFDTD

TODO: Add build status

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

## 1D FTDT

TODO: Explain formulas in 1D + results

The update equations in 1D are given as:
$\begin{align}
\mu \frac{\partial H_y}{\partial t} &= \frac{\partial E_z}{\partial x} \\
\epsilon \frac{\partial E_z}{\partial t} &= \frac{\partial H_y}{\partial x}
\end{align}$

## 2D FTDT

TODO: Explain formulas in 2D + results

The update equations in 1D are given as:


### $TM^z$
$\begin{align}
-\sigma_m H_x - \mu \frac{\partial H_x}{\partial t} &= \frac{\partial E_z}{\partial y} \\
\sigma_m H_y + \mu \frac{\partial H_y}{\partial t} &= \frac{\partial E_z}{\partial x} \\
\sigma E_z + \epsilon \frac{\partial E_z}{\partial t} &= \frac{\partial H_y}{\partial x} -\frac{\partial H_x}{\partial y}
\end{align}$

### $TE^z$

$\begin{align}
\sigma E_x + \epsilon \frac{\partial E_x}{\partial t} &= \frac{\partial H_z}{\partial y} \\
\sigma E_y + \epsilon \frac{\partial E_y}{\partial t} &= -\frac{\partial H_z}{\partial x} \\
-\sigma_m H_z - \mu\frac{\partial H_z}{\partial t} &= \frac{\partial E_y}{\partial x} - \frac{\partial E_x}{\partial y}
\end{align}$

## 3D FTDT

TODO: Explain formulas in 3D + results

## Testing

TODO: Explain what tests are performed

## Results and conclusions

TODO: Write some conclusion and to what extent the code can be extended

## References

[1] Understanding the Finite-Difference Time-Domain Method, John B. Schneider, [www.eecs.wsu.edu/~schneidj/ufdtd](www.eecs.wsu.edu/~schneidj/ufdtd), 2010. (also [here](./references/ufdtd.pdf))

[2] Berenger, Jean-Pierre. "A perfectly matched layer for the absorption of electromagnetic waves." Journal of computational physics 114.2 (1994): 185-200 (also [here](./references/APerfectlyMatchedLayerfortheAbsorptionofElectromagneticWaves.pdf))


TODO: Complete the list with useful references
