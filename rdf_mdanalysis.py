# this script uses the MD Analysis tools to compute RDFs.  MD Analysis should be loaded with following lines
#    module load anaconda3/latest
#    source activate p4env
#
from MDAnalysis import *
import MDAnalysis.analysis.rdf as rdf
import numpy as np

# set the atoms for computing RDF
resname1='BF4'
atoms1=('B',)
resname2='BMI'
atoms2=('CT',)

# frame to start, may want to skip frames for equilibration
framestart=1000

# set the pdb topology and dcd trajectory
topology="start_drudes.pdb"
trajectory="md_npt.dcd"


u=Universe(topology, trajectory)

print u.trajectory


group1 = u.select_atoms("name XXX")
group2 = u.select_atoms("name XXX")

for ele in atoms1:
  group1 = group1 + u.select_atoms("name %s and resname %s" % (ele, resname1) )

for ele in atoms2:
  group2 = group2 + u.select_atoms("name %s and resname %s" % (ele, resname2) )

rdf = rdf.InterRDF( group1 , group2 , nbins=400, range=(0.0,20.0), start=framestart )
rdf.run()

for i in range(len(rdf.bins)):
   print rdf.bins[i] , rdf.rdf[i]


