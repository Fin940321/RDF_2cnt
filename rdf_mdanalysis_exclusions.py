# this script uses the MD Analysis tools to compute RDFs.  MD Analysis should be loaded with following lines
#    module load anaconda3/latest
#    source activate p4env
#
from MDAnalysis import *
import MDAnalysis.analysis.rdf as rdf
import numpy as np

# this script is for like-like molecule RDFs, so we only have one set of atom types, and exclude intra-molecular pairs
# set the atoms for computing RDF
resname='BMI'
atoms=('C1','C2','C21')

# frame to start
framestart=500

# set the pdb topology and dcd trajectory
topology="eq_npt.pdb"
trajectory="md_nvt.dcd"


# this sets the intra-molecular exclusions (3,3) means three exclused atom types per molecule
exclusions=(len(atoms),len(atoms))


# create string to search for atoms, note all atoms in same molecule must be contiguous in list for exclusions to work
string=""
for i in range(len(atoms)):
    ele=atoms[i]
    string = string + "name %s and resname %s" % (ele, resname)
    if i < len(atoms)-1:
        string = string + " or "



u=Universe(topology, trajectory)

print u.trajectory

# select atoms, all atoms in molecule are adjacent in list
group =  u.select_atoms( string )

rdf = rdf.InterRDF( group , group , nbins=400, range=(0.0,20.0), exclusion_block=exclusions, start=framestart )
rdf.run()

for i in range(len(rdf.bins)):
   print rdf.bins[i] , rdf.rdf[i]


