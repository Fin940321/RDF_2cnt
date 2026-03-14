#!/usr/bin/env

from mdtraj import *
traj = mdtraj.load('md_nvt.dcd', top='start_drudes.pdb',stride=1)
top=traj.topology
pairs=top.select_pairs('name B','name B')
r,g_r = mdtraj.compute_rdf(traj,pairs,r_range=(0.0,4.0),periodic=True)

for i in range(len(r)):
    print( r[i] , g_r[i] )

