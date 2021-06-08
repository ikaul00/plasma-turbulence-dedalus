# Simulating Plasma Turbulence Using Dedalus
This page documents my journey through my work on simulating edge plasma turbulence using the [Dedalus](https://dedalus-project.org/) code under Prof. [Greg Hammett](https://w3.pppl.gov/~hammett/). This project is being done as part of my [PACM](https://www.pacm.princeton.edu/undergraduate) (Program in Applied and Computational Math) certificate project. The precursor to this work was a hydrodynamics code I developed as part of the APC 523 Numerical Algorithms class which can be found on github [here](https://github.com/ikaul00/finite-volume-Navier-Stokes). In particular this work aims to implement Hasegawa-Mima/Terry-Horton like equations using pseudo-spectral methods. For this project I am using a Google Colab jupyter notebook. It turns out the installation of Dedalus is non-trivial so we first begin with the installation steps.
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
The equations
