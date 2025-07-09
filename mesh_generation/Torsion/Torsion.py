from mpi4py import MPI
from dolfinx import mesh, fem, io, plot, nls, log, geometry, la
import basix.ufl
import dolfinx.fem.petsc
import ufl
import numpy as np
from petsc4py.PETSc import ScalarType
from petsc4py import PETSc
import time
import os
import gmsh


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# Parameters for the geometry


L = 5
Rad = 3
thickness = 0.15
R_in = Rad - thickness 


eps = 0.016
h = 0.015


make_mesh = True

if make_mesh:
    gmsh.initialize()
    if MPI.COMM_WORLD.rank == 0:
        gmsh.model.add("Torsion")


        # Create outer cylinder (along z-axis from z=0 to z=L)
        outer_cyl = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, L, Rad)

        # Create inner cylinder (same axis, smaller radius)
        inner_cyl = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, L, R_in)

        # Cut inner cylinder from outer cylinder to form a tube
        tube, _ = gmsh.model.occ.cut([(3, outer_cyl)], [(3, inner_cyl)])

        # Synchronize to reflect the changes in the model
        gmsh.model.occ.synchronize()

        # Add physical group for the volume (the tube itself)
        tube_volumes = [entity[1] for entity in tube]
        tube_group = gmsh.model.addPhysicalGroup(3, tube_volumes)
        gmsh.model.setPhysicalName(3, tube_group, "TubeVolume")


        # Define mesh size fields
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)

        # Generate and optimize the mesh
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.optimize("Netgen")
        gmsh.write("tube.msh")


    model = MPI.COMM_WORLD.bcast(gmsh.model, root=0)
    partitioner = dolfinx.cpp.mesh.create_cell_partitioner(dolfinx.mesh.GhostMode.shared_facet)
    mesh_data = io.gmshio.model_to_mesh(model, MPI.COMM_WORLD, 0, gdim=3, partitioner=partitioner)

    gmsh.finalize()
    MPI.COMM_WORLD.barrier()
    domain = mesh_data[0]