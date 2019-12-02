#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 10:24:38 2017

@author: jacob
"""

"""
Dedalus script for 2D SI simulations

inputs:
    N2  - background buoyancy frequency, units of s^-2
    theta - slope angle 
    ndays - number of days to simulate
    
    optionally 4 additional parameters setting domain size
    Lx  - Length of domain across-slope, units of m
    Lz  - Length of domain in slope normal, units of m
    nx  - Number of Fourier components in x direction
    nz  - Number of Chebyshev components in z direction
    
To run, merge, and plot using 4 processes, for instance, you could use:
    $ mpiexec -n 4 python3 2D_SI_DNS.py 1e-5 0.05 10
    $ mpiexec -n 4 python3 merge.py snapshots

"""

import numpy as np
from mpi4py import MPI
CW = MPI.COMM_WORLD
import time
from pylab import *
from dedalus import public as de
from dedalus.extras import flow_tools
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import logging
import sys
logger = logging.getLogger(__name__)


# Parameters

f = 1e-4 # Coriolis parameter
nu = 1e-4 # Background viscosity
kap = nu # Unit Pr
N = np.sqrt(float(sys.argv[1]))
rho = 1030
tht = float(sys.argv[2])
V = 0.1
ndays = float(sys.argv[3])
if len(sys.argv)>4:
    Lx = float(sys.argv[4])
    Lz = float(sys.argv[5])
    nx = int(sys.argv[6])
    nz = int(sys.argv[7])
    V = float(sys.argv[8])
else:
    nx = 1024
    nz = 256
    Lx,  Lz = (1000., 200.)
    
logger.info('===========PARAMETERS============')
logger.info(f'N = {N}, Theta = {tht}, Lx = {Lx}, Lz = {Lz}, Nx = {nx}, Nz = {nz}')

#%% 2D PROBLEM
# Create basis and domain
start_init_time = time.time()
x_basis = de.Fourier('x', nx, interval=(0, Lx), dealias=3/2)
z_basis = de.Chebyshev('z', nz, interval=(0, Lz), dealias=3/2)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64, mesh=None)
z = domain.grid(1)
x = domain.grid(0)

# Define Fields
damp = domain.new_field(name='damp') #Adding a sponge layer in the upper 20 meters
damp['g'] = -0.005*((-20 - (z-Lz))/20)**2
damp['g'][:,(z[0,:]-Lz)<-20] = 0
bb = domain.new_field(name='bb') # Complete background buoyancy field
bb['g']=N**2*z*np.cos(tht) + N**2*x*np.sin(tht)

# set up IVP
problem = de.IVP(domain, variables=['u', 'v', 'w', 'b', 'p', 'uz', 'vz', 'wz', 'bz'])
problem.meta[:]['z']['dirichlet'] = True
slices = domain.dist.grid_layout.slices(scales=1)

# define constants
problem.parameters['Lx'] = Lx
problem.parameters['Lz'] = Lz
problem.parameters['f'] = f
problem.parameters['kap'] = kap
problem.parameters['rho'] = rho
problem.parameters['bb'] = bb #Initial buoyancy field for damping (how to make 3D...?)
problem.parameters['damp'] = damp
problem.parameters['V'] = V # Initial constant barotropic flow (maintained throughout integration).
problem.parameters['N'] = N # Fixed background stratification (also sets across-slope gradient)
problem.parameters['tht'] = tht

# define substitutions
problem.substitutions['dy(A)'] = '0' #2D periodic, can remove for 3D
problem.substitutions['D(A, Az)'] = 'kap*(dz(Az)+dy(dy(A)) + dx(dx(A)))' # 3D diffusion operator
problem.substitutions['NL(A,Az)'] = 'u*dx(A) + v*dy(A) + w*Az' # nonlinear operator

#define substitions for diagnostics
problem.substitutions['wc'] = 'w*cos(tht) + u*sin(tht)' #True vert velocity
problem.substitutions['uc'] = 'u*cos(tht) - w*sin(tht)' #True u velocity
problem.substitutions['vc'] = 'v+V' # True v velocity
problem.substitutions['dzc(A)'] = 'dz(A)*cos(tht) + dx(A)*sin(tht)'
problem.substitutions['dxc(A)'] = 'dx(A)*cos(tht) - dz(A)*sin(tht)'

problem.substitutions['havg(A)'] = "integ(A, 'x')/Lx"
problem.substitutions['davg(A)'] = "integ(A, 'x', 'z')/(Lx*Lz)"
problem.substitutions['prime(A)'] = "A - havg(A)"
problem.substitutions['PV_vert'] = '(f+dxc(vc))*(N**2+dzc(b))'
problem.substitutions['PV'] = "PV_vert-(dzc(vc))*(dxc(b))"

# define equations
problem.add_equation('dt(u) - f*v*cos(tht) + dx(p)-b*sin(tht) +V*dy(u) - D(u, uz)   = -NL(u,uz)+damp*u')
problem.add_equation('dt(v) + f*u*cos(tht) -f*w*sin(tht) + dy(p) +V*dy(v)- D(v, vz)  = -NL(v,vz)+damp*v')
problem.add_equation('dt(w) + dz(p) - b*cos(tht) -D(w, wz) + V*dy(w) +f*v*sin(tht) = -NL(w, wz) + damp*w')
problem.add_equation('dt(b) + u*N**2*sin(tht) + w*N**2*cos(tht)+ V*dy(w)  - D(b,bz)   = -NL(b,bz)+damp*(b)')
problem.add_equation('dx(u) + dy(v) + wz = 0')

# define derivatives
problem.add_equation('uz - dz(u) = 0')
problem.add_equation('vz - dz(v) = 0')
problem.add_equation('wz - dz(w) = 0')
problem.add_equation('bz - dz(b) = 0')

# define boundary conditions
problem.add_bc('left(u) = 0')
problem.add_bc('left(v) = -V')
problem.add_bc('left(w) = 0')
problem.add_bc('left(bz) = -N**2*cos(tht)')
problem.add_bc('right(uz) = 0')
problem.add_bc('right(vz) = 0')
problem.add_bc('right(w) = 0', condition='(nx != 0)')
problem.add_bc('left(p) = 0', condition='(nx == 0)')
problem.add_bc('right(bz) = 0')

# Build solver
solver = problem.build_solver(de.timesteppers.RK443)
logger.info('Solver built')

# Initial conditions
b = solver.state['b']
u = solver.state['u']
v = solver.state['v']
bz = solver.state['bz']
uz = solver.state['uz']
vz = solver.state['vz']

zt = z
b['g'] = 0
u['g'] = 0
v['g'] = 0

# Random perturbations, initialized globally for same results in parallel
gshape = domain.dist.grid_layout.global_shape(scales=1)
slices = domain.dist.grid_layout.slices(scales=1)
rand = np.random.RandomState(seed=23)
noise = rand.standard_normal(gshape)[slices]

b['g'] += nx*1e-10*noise

# Calculate Derivatives
b.differentiate('z', out=bz)
u.differentiate('z', out=uz)
v.differentiate('z', out=vz)

#%%
# Integration parameters
solver.stop_sim_time = 3600*24*ndays
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

# Analysis
snap = solver.evaluator.add_file_handler('snapshots', sim_dt=1800, max_writes=24*10000)


# Basic fields
snap.add_task("u", scales=1, name='u')
snap.add_task("v", scales=1, name='v')
snap.add_task("w", scales=1, name='w')
snap.add_task("b", scales=1, name='b')

# PV Diagnostics
snap.add_task('PV', scales=1, name='PV')
snap.add_task("davg(PV)", scales=1, name='PVa') # Domain averaged PV
snap.add_task('havg(prime(w)*prime(PV))', scales=1, name='wq') # Vertical PV flux

# Energy diagnostics
snap.add_task('havg(prime(wc)*prime(b+bb))', scales=1, name='VBF')
snap.add_task('havg(prime(w)*prime(b))', scales=1, name='VBFrotated')
snap.add_task('havg(prime(u)*prime(b))', scales=1, name='HBFrotated')
# snap.add_task('havg(prime(wc)*kap*dzc(dzc(b+bb)))', scales=1, name='VBFsg')
snap.add_task('havg(prime(w)*prime(b)*cos(tht) + prime(u)*prime(b)*sin(tht))', scales=1, name='VBFr')
snap.add_task('havg(prime(vc)*prime(wc))*havg(dxc(b))/f', scales=1, name='GSP')#assumes background buoyancy has no x-variations (ie N**2 only).
snap.add_task('havg(prime(u)*prime(w))', scales=1, name='UPWP')

snap.add_task('havg(prime(vc)*prime(wc))*havg(dzc(vc))', scales=1, name='VSPv')
snap.add_task('havg(prime(uc)*prime(wc))*havg(dzc(uc))', scales=1, name='VSPu')
snap.add_task('havg(prime(vc)*prime(uc))*havg(dxc(vc))', scales=1, name='LSPv')
snap.add_task('havg(prime(uc)*prime(uc))*havg(dxc(uc))', scales=1, name='LSPu')
snap.add_task('havg(prime(v)*prime(w)*havg(vz) + prime(u)*prime(w)*havg(uz))', scales=1, name='SP')
snap.add_task('havg(prime(v)*prime(w))*N**2*sin(tht)/f', scales=1, name = 'GSProtated')

snap.add_task('havg(prime(u)*D(prime(u), prime(uz))) + havg(prime(v)*D(prime(v), prime(vz)) + havg(w)*D(prime(w), prime(wz)))', scales=1, name='Diss')
snap.add_task('havg(kap*(prime(uz)**2 + prime(vz)**2 + prime(wz)**2 + dx(prime(u))**2 + dx(prime(v))**2 + dx(prime(w))**2))', scales=1, name='DissPartial')

snap.add_task('0.5*havg(prime(u)**2 + prime(v)**2 + prime(w)**2)', scales=1, name='EKE')
snap.add_task('0.5*havg(prime(b)**2/N**2)', scales=1, name='EPE')
snap.add_task('havg(prime(b)*D(prime(b), prime(bz)))/N**2', scales=1, name='DISSBPRIME')
snap.add_task('kap*havg(dx(prime(b))**2 + dz(prime(b))**2)', scales=1, name='DISSBPARTIAL')
snap.add_task('N', name = 'N')
snap.add_task('damp', name='damp')
snap.add_task('tht', name='tht')
snap.add_task('havg(dxc(b))', name='bxbar')

#Spectral Tasks
snap.add_task('vc', layout=domain.dist.layouts[1], scales=1, name ='vcs') # Save in horizontal wavenumber, vertical physical space
snap.add_task('wc', layout=domain.dist.layouts[1], scales=1, name ='wcs')

snap.add_task('v', layout=domain.dist.layouts[1], scales=1, name ='vs') # Save in horizontal wavenumber, vertical physical space
snap.add_task('w', layout=domain.dist.layouts[1], scales=1, name ='ws')
snap.add_task('u', layout=domain.dist.layouts[1], scales=1, name ='us') # Save in horizontal wavenumber, vertical physical space
snap.add_task('b', layout=domain.dist.layouts[1], scales=1, name ='bs')

snap.add_task('NL(u, uz)', layout=domain.dist.layouts[1], scales=1, name='NLu')
snap.add_task('NL(v, vz)', layout=domain.dist.layouts[1], scales=1, name='NLv')
snap.add_task('NL(w, wz)', layout=domain.dist.layouts[1], scales=1, name='NLw')

# CFL
CFL = flow_tools.CFL(solver, initial_dt=1e-4, cadence=10, safety=0.8,
                     max_change=1.5, min_change=0, max_dt=30, threshold=0.05) #Adjust these for your problem
CFL.add_velocities(('u', 'w'))

# Flow properties

# Main loop
end_init_time = time.time()
logger.info('Initialization time: %f' %(end_init_time-start_init_time))
try:
    logger.info('Starting loop')
    start_run_time = time.time()
    while solver.ok:
        dt = CFL.compute_dt()
        solver.step(dt)
        if (solver.iteration-1) % 30 == 1:
            logger.info('Iteration: %i, Days: %1.2f, dt: %e' %(solver.iteration, solver.sim_time/86400, dt))
            utemp = solver.state['u']
            if utemp['g'].size > 0:
                um = np.max(np.abs(utemp['g']))
                logger.info('U Val: %f' % um)
                if np.isnan(um):
                    raise Exception('NaN encountered.') # This is here to catch model blow up
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_run_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_run_time-start_run_time))
    logger.info('Run time: %f cpu-hr' %((end_run_time-start_run_time)/60/60*domain.dist.comm_cart.size))
