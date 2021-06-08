# Simulating Plasma Turbulence Using Dedalus
This page documents my journey through my work on simulating edge plasma turbulence using the [Dedalus](https://dedalus-project.org/) code under Prof. [Greg Hammett](https://w3.pppl.gov/~hammett/). This project is being done as part of my [PACM](https://www.pacm.princeton.edu/undergraduate) (Program in Applied and Computational Math) certificate project. The precursor to this work was a hydrodynamics code I developed as part of the [APC 523](https://registrar.princeton.edu/course-offerings/course-details?term=1214&courseid=009654) Numerical Algorithms class which can be found on github [here](https://github.com/ikaul00/finite-volume-Navier-Stokes). In particular this work aims to implement Hasegawa-Mima/Terry-Horton like equations using pseudo-spectral methods. For this project I am using a Google Colab jupyter notebook. It turns out the installation of Dedalus is non-trivial so we first begin with the installation steps.
## Google Colab Installation

Step 1: Install the FFTW libraries
```
!apt-get install libfftw3-dev
!apt-get install libfftw3-mpi-dev
```
Step 2: Set paths for Dedalus installation on colab
```
import os
os.environ['MPI_INCLUDE_PATH'] = "/usr/lib/x86_64-linux-gnu/openmpi/include"
os.environ['MPI_LIBRARY_PATH'] = "/usr/lib/x86_64-linux-gnu"
os.environ['FFTW_INCLUDE_PATH'] = "/usr/include"
os.environ['FFTW_LIBRARY_PATH'] = "/usr/lib/x86_64-linux-gn"
```
Step 3: Install dedalus
```
!pip3 install dedalus
```
This should get dedalus up and running on Colab.

## Equations
The 2D neutral fluid Navier Stokes equation(NSE) (in absence of gravity) is given as\
<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;\vec{v}}{\partial&space;t}&space;&plus;&space;(\vec{v}&space;\cdot&space;\nabla)\vec{v}&space;=&space;\nu&space;\nabla^2\vec{v}&space;-&space;\nabla&space;p" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;\vec{v}}{\partial&space;t}&space;&plus;&space;(\vec{v}&space;\cdot&space;\nabla)\vec{v}&space;=&space;\nu&space;\nabla^2\vec{v}&space;-&space;\nabla&space;p" title="\frac{\partial \vec{v}}{\partial t} + (\vec{v} \cdot \nabla)\vec{v} = \nu \nabla^2\vec{v} - \nabla p" /></a>\
We consider this case along with the incompressibility condition <a href="https://www.codecogs.com/eqnedit.php?latex=\nabla\cdot~\vec{v}~&space;=~&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\nabla\cdot~\vec{v}~&space;=~&space;0" title="\nabla\cdot~\vec{v}~ =~ 0" /></a> and defining two new terms. the vorticity <a href="https://www.codecogs.com/eqnedit.php?latex=\omega&space;=&space;(\nabla\times\vec{v})\cdot\hat{z}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\omega&space;=&space;(\nabla\times\vec{v})\cdot\hat{z}" title="\omega = (\nabla\times\vec{v})\cdot\hat{z}" /></a> and the stream potential <a href="https://www.codecogs.com/eqnedit.php?latex=\psi:&space;\nabla\psi\times\hat{z}&space;=&space;\vec{v}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\psi:&space;\nabla\psi\times\hat{z}&space;=&space;\vec{v}" title="\psi: \nabla\psi\times\hat{z} = \vec{v}" /></a>

With these we cast the NSE into the vorticity formulation as follows:\

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial\omega}{\partial&space;t}&space;=&space;-[\omega,&space;\psi]&space;&plus;&space;\nu\nabla^2\omega" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial\omega}{\partial&space;t}&space;=&space;-[\omega,&space;\psi]&space;&plus;&space;\nu\nabla^2\omega" title="\frac{\partial\omega}{\partial t} = -[\omega, \psi] + \nu\nabla^2\omega" /></a>\
along with the Poisson equation\
<a href="https://www.codecogs.com/eqnedit.php?latex=\nabla^2\psi&space;=&space;-\omega" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\nabla^2\psi&space;=&space;-\omega" title="\nabla^2\psi = -\omega" /></a>\
