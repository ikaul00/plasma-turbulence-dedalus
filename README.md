# Simulating Plasma Turbulence Using Dedalus
This page documents my journey through my work on simulating plasma turbulence in a tokamak using the [DEDALUS](https://dedalus-project.org/) code under Prof. [Greg Hammett](https://w3.pppl.gov/~hammett/). This project is being done as part of my [PACM](https://www.pacm.princeton.edu/undergraduate) (Program in Applied and Computational Math) certificate project. The precursor to this work was a Finite Volume hydrodynamics code I developed as part of the [APC 523](https://registrar.princeton.edu/course-offerings/course-details?term=1214&courseid=009654) Numerical Algorithms class which can be found on github [here](https://github.com/ikaul00/finite-volume-Navier-Stokes). In particular this work aims to implement the Hasegawa-Mima, Terry-Horton and modified Terry-Horton equations using pseudo-spectral methods to model turbulence in a 2D vertical slice of a tokamak. The preliminary equations for hydrodynamics and the 2D vortex merger test case were adopted from [this paper](https://epubs.siam.org/doi/10.1137/120888053) by Peterson & Hammett. We then use the equations for the HME and THE as presented in previous papers: [this is one](https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/theory-of-the-tertiary-instability-and-the-dimits-shift-within-a-scalar-model/BAE474B82E6B5AD17FA6D9D10111C5FF), [this is another](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.124.055002) by [Zhu, H](https://theory.pppl.gov/people/profile.php?pid=115&n=Hongxuan-Zhu%E6%9C%B1%E9%B8%BF%E8%BD%A9),  [Dodin, I. Y.](https://theory.pppl.gov/people/profile.php?pid=21&n=Ilya-Dodin) and others. Our new insight is to adopt the shielding factor into the modified THE to better model turbulence in tokamaks. A discussion of the shielding factor and what it exactly is can be found in this [paper](https://aip.scitation.org/doi/10.1063/1.2972160) by my advisor and collaborators. It is known that vertical shears, called zonal flows, get produced during this turbulence, called drift wave turbulence, driven by the ion density gradient. We study the interplay of zonal flows and drift waves by varying the shielding parameter. We will later observe a well known result called the Dimits shift where the zonal flows dominate over drift waves below a certain density gradient threshold as demonstrated in this highly cited [paper](https://aip.scitation.org/doi/10.1063/1.873896) by Dimits and others. We will also show that the shielding factor suppresses zonal flows as it increases. Some numerics background: the engineering analysis [book](https://www.cambridge.org/core/books/fundamentals-of-engineering-numerical-analysis/D6B6B75172AD7A5A555DC506FDDA9B99#) by Moin contains a good overview of discrete transform/spectral methods (which is what DEDALUS uses, to be exact it uses pseudo spectral methods) in chapter 6. DEDALUS uses the Lancszos-tau method i.e., it solves the perturbed PDE exactly to find a close solution to the original PDE. However, it modifies it by not using a Chebyshev test function to produce sparse matrices. A complete description can be found in the DEDALUS methods [paper](https://doi.org/10.1103/PhysRevResearch.2.023068).

For this project I am using a Google Colab jupyter notebook. It turns out the installation of DEDALUS is slightly non-trivial so we first begin with the installation steps. (Skip to the subsequent section if you have a working DEDALUS installation)
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

## Test Case 1 
We will first show a test case for a vortex merger to check that our code works properly. Following this we will employ the plasma equations for turbulence. But first the basic hydrodynamc case. The 2D neutral fluid Navier Stokes equation(NSE) (in absence of gravity) is given as\
<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;\vec{v}}{\partial&space;t}&space;&plus;&space;(\vec{v}&space;\cdot&space;\nabla)\vec{v}&space;=&space;\nu&space;\nabla^2\vec{v}&space;-&space;\nabla&space;p" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;\vec{v}}{\partial&space;t}&space;&plus;&space;(\vec{v}&space;\cdot&space;\nabla)\vec{v}&space;=&space;\nu&space;\nabla^2\vec{v}&space;-&space;\nabla&space;p" title="\frac{\partial \vec{v}}{\partial t} + (\vec{v} \cdot \nabla)\vec{v} = \nu \nabla^2\vec{v} - \nabla p" /></a>\
We consider this case along with the incompressibility condition <a href="https://www.codecogs.com/eqnedit.php?latex=\nabla\cdot~\vec{v}~&space;=~&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\nabla\cdot~\vec{v}~&space;=~&space;0" title="\nabla\cdot~\vec{v}~ =~ 0" /></a> and defining two new terms. the vorticity <a href="https://www.codecogs.com/eqnedit.php?latex=\omega&space;=&space;(\nabla\times\vec{v})\cdot\hat{z}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\omega&space;=&space;(\nabla\times\vec{v})\cdot\hat{z}" title="\omega = (\nabla\times\vec{v})\cdot\hat{z}" /></a> and the stream potential <a href="https://www.codecogs.com/eqnedit.php?latex=\psi:&space;\nabla\psi\times\hat{z}&space;=&space;\vec{v}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\psi:&space;\nabla\psi\times\hat{z}&space;=&space;\vec{v}" title="\psi: \nabla\psi\times\hat{z} = \vec{v}" /></a>

With these we cast the NSE into the vorticity formulation as follows:

<img src="https://latex.codecogs.com/svg.image?\frac{\partial&space;\omega}{\partial&space;t}&space;&plus;&space;[\omega,\psi]&space;=&space;\nu&space;\nabla^2\omega" title="\frac{\partial \omega}{\partial t} + [\omega,\psi] = \nu \nabla^2\omega" />

where the Poisson bracket has the usual definition

<img src="https://latex.codecogs.com/svg.image?[\omega,\psi]&space;=&space;\frac{\partial&space;\omega}{\partial&space;x}\frac{\partial&space;\psi}{\partial&space;y}&space;-&space;\frac{\partial&space;\omega}{\partial&space;y}\frac{\partial&space;\psi}{\partial&space;x}&space;" title="[\omega,\psi] = \frac{\partial \omega}{\partial x}\frac{\partial \psi}{\partial y} - \frac{\partial \omega}{\partial y}\frac{\partial \psi}{\partial x} " />

along with the Poisson equation 

<img src="https://latex.codecogs.com/svg.image?\nabla^2\psi&space;=-\omega" title="\nabla^2\psi =-\omega" />

### Dedalus Setup
Import dedalus and setup Fourier basis in x and y directions. Currently, dedalus allows to use a Chebyshev basis in one of the directions (which can be useful in problems with flow in a single direction like a KH instability problem, for example). This code was adopted to our problem with the help of the extensive [documentation](https://dedalus-project.readthedocs.io/en/latest/index.html) which contains many [tutorials](https://dedalus-project.readthedocs.io/en/latest/pages/tutorials.html).
```
# Import Statements
import numpy as np
import matplotlib.pyplot as plt                             

from dedalus import public as de

# Aspect ratio 1                                                                                                                                                     Lx, Ly = (2., 2.)
nx, ny = (200, 200)

# Create bases and domain 
x_basis = de.Fourier('x', nx, interval=(-Lx/2, Lx/2), dealias=1)
y_basis = de.Fourier('y', ny, interval=(-Ly/2, Ly/2), dealias=1)                                                                                                                                                           
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)
```

Now add the equations and parameters of the problem
```
viscosity = 0
problem = de.IVP(domain, variables=['omega','psi', 'modv'])
problem.parameters['nu'] = viscosity
problem.add_equation("dt(omega) -nu*dx(dx(omega)) - nu*dy(dy(omega))= dy(omega)*dx(psi) - dx(omega)*dy(psi) ")
problem.add_equation("-omega-dx(dx(psi)) - dy(dy(psi)) = 0", condition="(nx != 0) or (ny != 0)")
problem.add_equation("psi= 0", condition="(nx == 0) and (ny == 0)")
#problem.add_equation("modv = dy(psi)*dy(psi) + dx(psi)*dy(psi)")
```
For now, I have initialized the problem with no viscosity. Notice that the boundary conditions are taken care of automatically when we specify a Fourier basis: dedalus assumes they are double periodic by default. We also handle the case where the solution to the poisson equation gives infinity due to division by the squares of the frequecies in both dimensions when they are zero. This is implemeted using the 'condition' parameter. The last equation is not neccesary for the problem itself but is useful to analyze energy conservation.\
Next, specify time integrator and build the solver.
```
ts = de.timesteppers.RK443
solver =  problem.build_solver(ts)
```
Now we initialize our gaussian monopoles. As a test we also want to plot the energy and enstrophy evolution, which are initialized here.
```
# Intializing sigma and mu                                                                                                                                                                                                                          
sigma = 1
mu1 = 0.25
mu2=-0.25

# Initializing Gaussian monopoles on omega                                                                                                                                                                                                                      
dst1 = (x-mu1)**2 + (y)**2
dst2 = (x-mu2)**2 + (y)**2
omega['g'] = np.exp(-(dst1/(sigma**2 /20))) + np.exp(-(dst2 / (sigma**2 /20)))
#omega.differentiate('y', out = omegay)

solver.stop_sim_time = 25.01
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

dt = 0.01

# Enstrophy/Energy analysis, initialize here
enstrophy_arr = []
energy_arr = []
time_arr = []
enstrophy_arr.append(np.sum(np.array(omega['g'])*np.array(omega['g'])))
energy_arr.append(np.sum(np.array(u['g'])*np.array(u['g']) + np.array(v['g'])*np.array(v['g'])))


```

Finally, evolve the system
``` 
x = domain.grid(0,scales=domain.dealias)
y = domain.grid(1,scales=domain.dealias)
start_time = time.time()
t = 0
while solver.ok:
    #dt = cfl.compute_dt()
    solver.step(dt)
    if solver.iteration % 150 == 0:
        #print(omega['g'])
        # Update plot of scalar field                                                                                                                                                                                      
        enstrophy_arr.append(np.sum(np.array(omega['g'])*np.array(omega['g'])))
        energy_arr.append(np.sum(np.array(u['g'])*np.array(u['g']) + np.array(v['g'])*np.array(v['g'])))
        time_arr.append(t)
    t = t+dt
```
To plot the final snapshot of vorticity, use a simple plotting script
```
from mpl_toolkits.axes_grid1 import make_axes_locatable
fig, ax = plt.subplots(figsize=(6,6))
div = make_axes_locatable(ax)
cax = div.append_axes('right', '5%', '5%')
img = ax.imshow(omega['g'].T, origin='lower',cmap='viridis', extent = [-1,1,-1,1])
tx = ax.set_title('t = 25')
plt.gca().xaxis.tick_bottom()
plt.xlabel('x')
plt.ylabel('y')
fig.colorbar(img, cax = cax)
plt.savefig('plot_hydro.png', bbox_inches='tight', dpi = 200)
```
After which we finally get the following plot
<p float="left">
  <img src="plot_hydro.png" width="450" />
</p> 

And finally plot the energy and enstrophy evolution. These show that our code worked well, since the change in both of these quantities is very small. To further check your code, a smaller dt should yield better conservation properties, i.e. the change in energy and enstrophy should be smaller for smaller dt.
<p float="left">
  <img src="energy_cons_dt.png" width="330" />
  <img src="enstrophy_cons_dt.png" width="330" /> 
</p>
### Test 2
Now we move on to the Hasegawa-Mima equation (HME). The HME adds in a y drift term to the usual fluid equations.

<img src="https://latex.codecogs.com/svg.image?\frac{\partial&space;\xi}{\partial&space;t}&space;&plus;&space;\textbf{v}\cdot\nabla\xi&space;-&space;\kappa\frac{\partial&space;\xi}{\partial&space;y}&space;=&space;0" title="\frac{\partial \xi}{\partial t} + \textbf{v}\cdot\nabla\xi - \kappa\frac{\partial \xi}{\partial y} = 0" />

along with a slight modification in the Poisson equation

<img src="https://latex.codecogs.com/svg.image?\xi&space;=&space;\nabla^2\psi&space;-\psi" title="\xi = \nabla^2\psi -\psi" />

We modify the code based on the new equations.
```
viscosity = 0
k=5.5
problem = de.IVP(domain, variables=['omega','psi','u','v', 'u2', 'v2'])                                                                                                                                                                                        
problem.parameters['nu'] = viscosity
problem.parameters['k'] = k
problem.add_equation("dt(omega) -nu*dx(dx(omega)) - nu*dy(dy(omega)) - k*dy(psi)= dy(omega)*dx(psi) - dx(omega)*dy(psi)")
problem.add_equation("omega-dx(dx(psi)) - dy(dy(psi)) + psi = 0", condition="(nx != 0) or (ny != 0)")
problem.add_equation("psi= 0", condition="(nx == 0) and (ny == 0)")
problem.add_equation("u + dy(psi)=0")
problem.add_equation("v - dx(psi) = 0")
problem.add_equation("u2 + dy(u) = 0")
problem.add_equation("v2 - dx(v) = 0")
```

Similarly for the Terry-Horton Eqaution (THE) we introduce the modifications

<img src="https://latex.codecogs.com/svg.image?\frac{\partial&space;\xi}{\partial&space;t}&space;&plus;&space;\textbf{v}\cdot\nabla\xi&space;-&space;\kappa\frac{\partial&space;\xi}{\partial&space;y}&space;&plus;&space;\hat{D}\xi=&space;0" title="\frac{\partial \xi}{\partial t} + \textbf{v}\cdot\nabla\xi - \kappa\frac{\partial \xi}{\partial y} + \hat{D}\xi= 0" />

and more importantly in the Poisson equation

<img src="https://latex.codecogs.com/svg.image?\xi&space;=&space;\nabla^2\psi&space;-&space;(1-i\hat{\delta})\psi" title="\xi = \nabla^2\psi - (1-i\hat{\delta})\psi" />


```
viscosity = 0
k=5.5
delta0 = 1.5
alpha = 0.05
problem = de.IVP(domain, variables=['omega','psi', 'u','v', 'u2', 'v2'])                                                    
problem.parameters['nu'] = viscosity
problem.parameters['k'] = k
problem.parameters['d0'] = delta0
problem.parameters['alpha'] = alpha
               
problem.add_equation("dt(omega) -nu*dx(dx(omega)) - nu*dy(dy(omega)) + alpha*omega - k*dy(psi)= dy(omega)*dx(psi) - dx(omega)*dy(psi)")
problem.add_equation("omega-dx(dx(psi)) - dy(dy(psi)) + psi - d0*dy(psi)= 0", condition="(nx != 0) or (ny != 0)")
problem.add_equation("psi= 0", condition="(nx == 0) and (ny == 0)")
problem.add_equation("u + dy(psi)=0")
problem.add_equation("v - dx(psi) = 0")
problem.add_equation("u2 + dy(u) = 0")
problem.add_equation("v2 - dx(v) = 0")
```

Now we describe our modification to the THE, the MTHE along with neoclassical shielding. The evolution equation becomes

<img src="https://latex.codecogs.com/svg.image?\frac{\partial&space;\xi}{\partial&space;t}&space;&plus;&space;\textbf{v}\cdot\nabla\xi&space;-&space;\kappa\frac{\partial&space;\xi}{\partial&space;y}&space;&plus;&space;\hat{D}(\xi&space;-&space;\langle\xi\rangle)&space;=&space;0" title="\frac{\partial \xi}{\partial t} + \textbf{v}\cdot\nabla\xi - \kappa\frac{\partial \xi}{\partial y} + \hat{D}(\xi - \langle\xi\rangle) = 0" />

and the shielding factor comes in to the Poisson equation

<img src="https://latex.codecogs.com/svg.image?\xi&space;=&space;\nabla^2\psi&space;&plus;&space;(s-1)\nabla^2\langle\psi\rangle-&space;(1-i\delta_0&space;\frac{\partial}{\partial&space;y})\psi&space;&plus;\langle\psi\rangle" title="\xi = \nabla^2\psi + (s-1)\nabla^2\langle\psi\rangle- (1-i\delta_0 \frac{\partial}{\partial y})\psi +\langle\psi\rangle" />

where the angular brackets represent the y averaged quantity.
THe code gets modified to
```
viscosity = 0 #1e-2
k=6.0 #0.5
s=10.0
delta0 = 1.5
alpha = 0.0 #0.04
alphad = 1
nud = 0.01
problem = de.IVP(domain, variables=['omega','psi', 'u','v', 'shear_ZF', 'shear_DW', 'psi_DW'])
problem.parameters['nu'] = viscosity
problem.parameters['k'] = k
problem.parameters['d0'] = delta0
problem.parameters['alpha'] = alpha
problem.parameters['Ly'] = Ly
problem.parameters['Lx'] = Lx
problem.parameters['alphad'] = alphad
problem.parameters['nud'] = nud 
problem.parameters['s'] = s                                                                                                                                                                                       
problem.add_equation("dt(omega) -nu*dx(dx(omega)) - nu*dy(dy(omega)) + alpha*omega - k*dy(psi) +(omega-integ((omega),'y')/Ly)*alphad - nud*(dx(dx(omega-integ((omega),'y')/Ly))+ dy(dy(omega-integ((omega),'y')/Ly)))= dy(omega)*dx(psi) - dx(omega)*dy(psi)")
problem.add_equation("omega-dx(dx(psi)) -(s-1)*( dx(dx(integ((psi),'y')/Ly))+ dy(dy(integ((psi),'y')/Ly)) )- dy(dy(psi)) + psi - d0*dy(psi) - integ((psi),'y')/Ly= 0", condition="(nx != 0) or (ny != 0)")
problem.add_equation("psi= 0", condition="(nx == 0) and (ny == 0)")
problem.add_equation("u + dy(psi)=0")
problem.add_equation("v - dx(psi) = 0")
problem.add_equation("shear_ZF= (integ((dx(dx((integ((psi),'y')/Ly)))**2),'x')/Lx)**(1/2)")
problem.add_equation("psi_DW = psi - integ((psi),'y')/Ly")
problem.add_equation("shear_DW = (        integ(   (   (dx(dy(psi)))**2 + (1/2)*(dx(dx(psi_DW)) - dy(dy(psi_DW)))**2  )  ,'x'      )/Lx        )**(1/2)")

```
## Numerical Result
We show the ion guiding center density evolution of the HME first
<p float="left">
  <img src="HME_k_5.5_snap1-1.png" width="330" />
  <img src="HME_k_5.5_snap2-1.png" width="330" /> 
  <img src="HME_k_5.5_snap3-1.png" width="330" />
</p>


The following plots show the ion guiding center density snapshots of the evolution of the MTHE with neoclassical shielding. Here we have k=7, s=1
<p float="left">
  <img src="k_7_s_1_snap1-1.png" width="330" />
  <img src="k_7_s_1_snap2-1.png" width="330" /> 
  <img src="k_7_s_1_snap3-1.png" width="330" />
</p> 

Here we have k=7, s=10

<p float="left">
  <img src="k_7_s_10_snap1-1.png" width="330" />
  <img src="k_7_s_10_snap2-1.png" width="330" /> 
  <img src="k_7_s_10_snap3-1.png" width="330" />
</p> 
These show the clear difference in varying the shielding parameter while keeping the density gradient parameter fixed.
  
