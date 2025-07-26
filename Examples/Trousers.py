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


log.set_log_level(log.LogLevel.WARNING)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


Lambda = ScalarType(85.77)                                       # Lambda in MPa
mu = ScalarType(0.52)                                            # Shear modulus in MPa


mu1 = mu
mu2 = ScalarType(0)
mu_pr = Lambda
alph1 = ScalarType(1)
alph2 = ScalarType(1)


Gc= ScalarType(0.041)	                                         #Critical energy release rate


sts, shs = 0.3, 1
Wts = 0.0373087
Whs = 0.005818


#The regularization length
eps = 0.21                                                                           
h = 0.05

delta = (1+3*h/(8*eps))**(-2) * ((sts + (1+2*np.sqrt(3))*shs)/((8+3*np.sqrt(3))*shs)) * 3*Gc/(16*Wts*eps) + (1+3*h/(8*eps))**(-1) * (2/5)



# Geometry of the treousers test
# Parameters

c_r = 5
real_cracksize = 48
L = 100  # Length of the rectangle
thickness = 1   # Thickness
H = 40    
ac = 50
A = real_cracksize - np.pi*c_r/2
B = L - real_cracksize
distance = 2*(A) + thickness + 2*c_r

dcrack = 2

print(distance)


partitioner = dolfinx.cpp.mesh.create_cell_partitioner(dolfinx.mesh.GhostMode.shared_facet)
mesh_data = io.gmshio.read_from_msh("trousers.msh", MPI.COMM_WORLD, gdim=3, partitioner=partitioner)
domain = mesh_data[0]
domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim)
domain.topology.create_entities(1)
domain.topology.create_entities(domain.topology.dim-1)
domain.topology.create_connectivity(domain.topology.dim-1, domain.topology.dim)
domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim-1)





def freeEnergy(u):
    """Calculates the free energy.

    Given `u`, this function calculates free energy.

    Args:
        u (dolfin.Function): FE displacement field.

    Returns:
        ufl.algebra.Sum or ufl.core.expr.Expr: The requested free energy.
    """
    F = (ufl.Identity(len(u)) + ufl.grad(u))
    C = (F.T * F)
    J = (ufl.det(F))
    I1 = (ufl.tr(C))


    psiEq = 3**(1-alph1)/(2.*alph1) * mu1 * (I1**alph1 - 3**alph1) + 3**(1-alph2) / \
        (2.*alph2) * mu2 * (I1**alph2 - 3**alph2) -(mu1 + mu2) * ufl.ln(abs(J)) + mu_pr/2 * (J - 1)**2
    return psiEq




def stressPiola(u, stress_type='first'):
    """[summary]
        Given `u` this function calculates
        the first Piola-Kirchhoff stress
    Args:
        u ([dolfin.Function]): [FE displacement field]

    Returns:
        [SEq]: [First PK stress tensor
        of type `ufl.algebra.Sum`]
    """
    F = (ufl.Identity(len(u)) + ufl.grad(u))
    C = (F.T * F)
    J = (ufl.det(F))
    Finv = (ufl.inv(F))
    I1 = (ufl.tr(C))



    SEq = (mu1 * (I1/3.)**(alph1 - 1) + mu2 * (I1/3.)**(alph2 - 1)) * \
        F - (mu1 + mu2) * Finv.T + mu_pr * J * (J-1.) * Finv.T

    if stress_type == 'second':
        return Finv * (SEq)
    else:
        return SEq



def local_project(v, V):
    """[summary]
        Helper function to do a interpolation
    Args:
        v ([dolfin.Funcion]): [function to be projected]
        V ([dolfin.Function]): [target `dolfin.FunctionSpace` to be projected on]

    Returns:
        [dolfin.Function]: [target function after projection]
    """
    expr = fem.Expression(v, V.element.interpolation_points, comm)
    u = fem.Function(V)
    u.interpolate(expr)
    return u

def adjust_array_shape(input_array):
    if input_array.shape == (2,):                                                 # Check if the shape is (2,)
        adjusted_array = np.append(input_array, 0.0)                              # Append 0.0 to the array
        return adjusted_array
    else:
        return input_array

bb_tree = geometry.bb_tree(domain, domain.topology.dim)


def evaluate_function(u, x):
    """Evaluates a function at a point `x` in parallel using MPI

    Args:
        u (dolfin.Function): Function to be evaluated
        x (Union(tuple, list, numpy.ndarray)): Point at which to evaluate function `u`

    Returns:
        numpy.ndarray: Function evaluated at point `x`
    """

    if isinstance(x, np.ndarray):
        # If x is already a NumPy array
        points0 = x
    elif isinstance(x, (tuple, list)):
        # If x is a tuple or list, convert it to a NumPy array
        points0 = np.array(x)
    else:
        # Handle the case if x is of an unsupported type
        points0 = None

    points = adjust_array_shape(points0)

    u_value_local = None

    cells = []
    # Find cells whose bounding-box collide with the points
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    # Choose one of the cells that contains the point
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)

    if len(colliding_cells.links(0)) > 0:
        u_value_local = u.eval(points, colliding_cells.links(0)[0])

    # Gather results from all processes
    u_value_all = comm.gather(u_value_local, root=0)

    # Concatenate results on root process
    if rank == 0:
        # Filter out None values before concatenation
        u_value_all = [arr for arr in u_value_all if arr is not None]
        # Concatenate arrays only if there are non-None values and if there's at least one valid value
        if u_value_all:
            u_value = np.concatenate(u_value_all[:1])  # Take the first valid value
        else:
            u_value = None
    else:
        u_value = None

    # Broadcast the final result to all processes
    u_value = comm.bcast(u_value, root=0)

    return u_value

def principal_invariants(u, z):
    """Calculates the first and second principal invariants of the second PK stress tensor.

    Given `u` and `z`, this function calculates the first Piola-Kirchhoff stress
    tensor from both equilibrium and non-equilibrium states, and returns the
    first and second principal invariants of the stress tensor.

    Args:
        u (dolfin.Function): FE displacement field.
        z (dolfin.Function): FE internal variable.

    Returns:
        tuple:
            - pi1 (ufl.algebra.Sum): First principal invariant of the stress tensor.
            - pi2 (ufl.algebra.Sum): Second principal invariant of the stress tensor.
    """
    F = (ufl.Identity(len(u)) + ufl.grad(u))
    C = (F.T * F)
    J = (ufl.det(F))
    I1 = (ufl.tr(C))
    I2 = (I1**2 - ufl.inner(C,C)) / 2


    xi1 = ((2**5)/27)*(2*I1**3 - 9*I1*I2 + 27*(J**2))
    xi2 = ((2**10)/27)*(4*I2**3 - (I1*I2)**2 + 4*(I1**3)*(J**2) - 18*I1*I2*J**2 + 27*J**4)
    xi3 = -2/3*I1 + (abs(xi1+ufl.sqrt(abs(xi2))))**(1/3) + (abs(xi1-ufl.sqrt(abs(xi2))))**(1/3)

    i1 = (ufl.sqrt(abs(2*I1 + xi3)) + \
          ufl.sqrt(abs(2*I1 - xi3 + (16*J)/(ufl.sqrt(abs(2*I1 + xi3))))))/2 
    i2 = ufl.sqrt(abs(I2 + 2*i1*J))

    psiEq_I1 = (mu1 * (I1/3.)**(alph1 - 1) + mu2 * (I1/3.)**(alph2 - 1))/2
    psiEq_J = - (mu1 + mu2) / J + mu_pr * (J-1.)


    pi1 = 2*z**2*i1*psiEq_I1 + z**2*i2*psiEq_J

    pi2 = 4*z**4*i2*psiEq_I1**2 + z**4*i1*J*psiEq_J**2 \
    + 2*z**4*(i1*i2 - 3*J)*psiEq_J*psiEq_I1


    return pi1, pi2

def ce(u, z):
    """Calculates the driving force for the phase-field equation.

    Given the displacement field `u`, `z`, this function
    calculates the driving force (ce) in the phase-field equation.

    Args:
        u (dolfin.Function): FE displacement field.
        z (dolfin.Function): FE phase field.

    Returns:
        ufl.core.expr.Expr: Driving force for the phase-field equation.
    """


    pi1, pi2 = principal_invariants(u, z)
    beta1=-(1/shs)*(delta)*Gc/(8*eps) + 2*Whs/(3*shs)
    beta2=-np.sqrt(3)*(3*shs-sts)/(shs*sts)*(delta)*Gc/(8*eps) - 2*Whs/(3**0.5*shs) + 2*3**0.5*Wts/sts


    output = beta2* ufl.sqrt(abs(pi1**2 / 3 - pi2)) + beta1*pi1 + z*(1 - ufl.sqrt(pi1**2)/pi1)*(freeEnergy(u))
    return output


def build_nullspace(V):
    """Build PETSc nullspace for 3D elasticity"""

    # Create vectors that will span the nullspace
    bs = V.dofmap.index_map_bs
    length0 = V.dofmap.index_map.size_local
    basis = [la.vector(V.dofmap.index_map, bs=bs, dtype=PETSc.ScalarType) for i in range(6)]
    b = [b.array for b in basis]

    # Get dof indices for each subspace (x, y and z dofs)
    dofs = [V.sub(i).dofmap.list.flatten() for i in range(3)]

    # Set the three translational rigid body modes
    for i in range(3):
        b[i][dofs[i]] = 1.0

    # Set the three rotational rigid body modes
    x = V.tabulate_dof_coordinates()
    dofs_block = V.dofmap.list.flatten()
    x0, x1, x2 = x[dofs_block, 0], x[dofs_block, 1], x[dofs_block, 2]
    b[3][dofs[0]] = -x1
    b[3][dofs[1]] = x0
    b[4][dofs[0]] = x2
    b[4][dofs[2]] = -x0
    b[5][dofs[2]] = x1
    b[5][dofs[1]] = -x2

    la.orthonormalize(basis)

    basis_petsc = [
        PETSc.Vec().createWithArray(x[: bs * length0], bsize=3, comm=V.mesh.comm)  # type: ignore
        for x in b
    ]
    return PETSc.NullSpace().create(vectors=basis_petsc)  # type: ignore


# Defining the function spaces
hel=ufl.FacetArea(domain)                                                         #area/length of a cell facet on a given mesh
h_avg = (hel('+') + hel('-')) / 2.0
n=ufl.FacetNormal(domain)

V = fem.functionspace(domain, ("CR", 1, (domain.geometry.dim,)))                  #Function space for u

Vplot = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))        #Function space for plotting u
Y = fem.functionspace(domain, ("Lagrange", 1))                                    #Function space for z

########################################################################
# Define functions
########################################################################

u = fem.Function(V)
un = fem.Function(V, name="displacement")                                         # displacement from previous iteration
u_inc = fem.Function(V, name="displacement-inc")                                  # Incremental displacement for solver
utrial = ufl.TrialFunction(V)                                                     # Incremental displacement
delu = ufl.TestFunction(V)                                                        # Test function for u
uiter = fem.Function(V, name='uk_1')                                              # Iteration variable for u
u_prev_iter = fem.Function(V)                                                     # Previous iteration for u
error_u = fem.Function(V)                                                         # Change in displacement from previous iteration
uplot = fem.Function(Vplot, name="displacement")                                  # Function for plotting u

z  = fem.Function(Y, name="Phase-field")                                          # Phase field
zn = fem.Function(Y, name="Phase-field")                                          # Phase field from previous iteration
z_inc = fem.Function(Y, name="Phase-field-inc")                                   # Incremental phase field for solver
ztrial = ufl.TrialFunction(Y)                                                     # Incremental phase field
delz  = ufl.TestFunction(Y)                                                       # Test function   Y before
z_prev_iter = fem.Function(Y)                                                     # Previous iteration for z
error_z = fem.Function(Y)                                                         # Change in phase field from previous iteration



def bottom(x):
    return np.isclose(x[1], 0)

def leftside(x):
    return np.isclose(x[0], -c_r - A)

def rightside(x):
    return np.isclose(x[0], c_r + thickness + A)

# Locate facets for boundary conditions

fdim = domain.topology.dim -1

bottom_facets = mesh.locate_entities_boundary(domain, fdim, bottom)
left_facets = mesh.locate_entities_boundary(domain, fdim, leftside)
right_facets = mesh.locate_entities_boundary(domain, fdim, rightside)


dofs_bottom0 = fem.locate_dofs_topological(V.sub(0), fdim, bottom_facets)
dofs_bottom1 = fem.locate_dofs_topological(V.sub(1), fdim, bottom_facets)
dofs_bottom2 = fem.locate_dofs_topological(V.sub(2), fdim, bottom_facets)

dofs_left0 = fem.locate_dofs_topological(V.sub(0), fdim, left_facets)
dofs_left1 = fem.locate_dofs_topological(V.sub(1), fdim, left_facets)
dofs_left2 = fem.locate_dofs_topological(V.sub(2), fdim, left_facets)

dofs_right0 = fem.locate_dofs_topological(V.sub(0), fdim, right_facets)
dofs_right1 = fem.locate_dofs_topological(V.sub(1), fdim, right_facets)
dofs_right2 = fem.locate_dofs_topological(V.sub(2), fdim, right_facets)




bcb0 = fem.dirichletbc(ScalarType(0), dofs_bottom0, V.sub(0))
bcb1 = fem.dirichletbc(ScalarType(0), dofs_bottom1, V.sub(1))
bcb2 = fem.dirichletbc(ScalarType(0), dofs_bottom2, V.sub(2))

bcl0 = fem.dirichletbc(ScalarType(0), dofs_left0, V.sub(0))
bcl1 = fem.dirichletbc(ScalarType(0), dofs_left1, V.sub(1))
bcl2 = fem.dirichletbc(ScalarType(0), dofs_left2, V.sub(2))

bcr0 = fem.dirichletbc(ScalarType(0), dofs_right0, V.sub(0))
bcr1 = fem.dirichletbc(ScalarType(0), dofs_right1, V.sub(1))
bcr2 = fem.dirichletbc(ScalarType(0), dofs_right2, V.sub(2))


bcs = [bcl0, bcl1, bcl2, bcr0, bcr1, bcr2] 




domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim)


def cracktip(x):
    """Checks if points lie in the crack tip region."""
    return np.logical_and.reduce((
        np.logical_and(-1e-4 - thickness  < x[0], x[0] < 2*thickness + 1e-4),
        np.logical_and(B - 1e-4 -dcrack   < x[1], x[1] < B -dcrack + 1e-4),
        np.logical_and(-H / 2 - h/2 - 1e-4 < x[2], x[2] < -H / 2 + h/2+1e-4)
    ))

cracktip_facets = mesh.locate_entities_boundary(domain, 0, cracktip)

dofs_cracktip= fem.locate_dofs_topological(Y, 0, cracktip_facets)

bct_z = fem.dirichletbc(ScalarType(0), dofs_cracktip, Y)
bcs_z = [bct_z]

marked_facets = np.hstack([left_facets, right_facets])
marked_values = np.hstack([np.full_like(left_facets, 1),
                           np.full_like(right_facets, 2)])
sorted_facets = np.argsort(marked_facets)
facet_tag = mesh.meshtags(domain, domain.topology.dim -1,
                          marked_facets[sorted_facets],
                          marked_values[sorted_facets])

metadata = {"quadrature_degree": 2}
ds = ufl.Measure('ds', domain=domain,
                 subdomain_data=facet_tag, metadata=metadata)
dS = ufl.Measure("dS", domain=domain, metadata=metadata)
dx = ufl.Measure("dx", domain=domain, metadata=metadata)

def norm_L2(comm, v):
    """Compute the L2(O)-norm of v"""
    return np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(v, v) * dx)), op=MPI.SUM))

# Initialize the fields

u.x.array[:] = 0.
u_prev_iter.x.array[:] = 0.


z.x.array[:] = 1.
z_prev_iter.x.array[:] = 1.



un.x.array[:] = u.x.array
zn.x.array[:] = z.x.array



qvals = (mu1+mu2)*1        # These two lines should be specified by the user for the problem at hand
eta = ScalarType(1e-9)

a_uv = ufl.derivative((z**2 + eta)*freeEnergy(u), u, delu)*dx \
+ qvals / h_avg * ufl.inner(ufl.jump(u), ufl.jump(delu)) * dS


Jac = ufl.derivative(a_uv, u, utrial)
Jac_imgrad = ufl.inner(ufl.grad(utrial),ufl.grad(delu))*dx

#Balance of configurational forces PDE
pen= 1000*(3*Gc/4/eps)*ufl.conditional(ufl.lt(delta,1),1, delta)
Wv = pen/2*((abs(z)-z)**2 + (abs(1-z) - (1-z))**2 )*dx
Wv2=ufl.conditional(ufl.le(z, 0.05), 1, 0)*100*pen/2*((1/4)*(abs(zn-z)-(zn-z))**2)*dx


a_z = delz*2*z*(freeEnergy(u))*dx - delz*(ce(u, z))*dx \
+ 3*delta*Gc/8*(-delz/eps + 2*eps*ufl.inner(ufl.grad(z),ufl.grad(delz)))*dx\
 + ufl.derivative(Wv,z,delz)+ ufl.derivative(Wv2,z,delz)

# Compute Jacobian of R_z
Jac_z = ufl.derivative(a_z, z, ztrial)


class NonlinearPDEProblem:
    """Nonlinear problem class for a PDE problem."""

    def __init__(self, F, J, u, bc):
        V = u.function_space
        du = ufl.TrialFunction(V)
        self.L = fem.form(F)
        self.a = fem.form(J)
        self.bc = bc
        self.internal_forces = None

    def form(self, x):
        from petsc4py import PETSc

        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def F(self, x, b):
        """Assemble residual vector."""
        from petsc4py import PETSc

        from dolfinx.fem.petsc import apply_lifting, assemble_vector, set_bc

        with b.localForm() as b_local:
            b_local.set(0.0)
        assemble_vector(b, self.L)
        # Store internal_forces before applying BCs
        self.internal_forces = b.copy()
        apply_lifting(b, [self.a], bcs=[self.bc], x0=[x], alpha=-1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, self.bc, x, -1.0)

    def J(self, x, A):
        """Assemble Jacobian matrix."""
        from dolfinx.fem.petsc import assemble_matrix

        A.zeroEntries()
        assemble_matrix(A, self.a, bcs=self.bc)
        A.assemble()

    def matrix(self):
        from dolfinx.fem.petsc import create_matrix

        return create_matrix(self.a)

    def vector(self):
        from dolfinx.fem.petsc import create_vector

        return create_vector(self.L)
    
    def get_reaction_forces(self, boundary_dofs):
        # Return the *negative* of the internal forces at the specified dofs
        if self.internal_forces is None:
            raise ValueError("Solve the problem first to compute internal forces.")

        b_array = fem.Function(V)
        b_array.x.array[:V.dofmap.index_map.size_local*V.dofmap.index_map_bs] = self.internal_forces.getArray()
        b_array.x.scatter_forward()
        return b_array.x.array[boundary_dofs]
    

class NonlinearPDEProblem_ig:
    """Nonlinear problem class for a PDE problem."""

    def __init__(self, F, J, Jim, u, bc, Omega=1e6):
        V = u.function_space
        du = ufl.TrialFunction(V)
        self.Omega = ScalarType(Omega)
        self.L = fem.form(F)
        self.a = fem.form(J + (1 / self.Omega) * Jim)
        self.bc = bc
        self.internal_forces = None

    def form(self, x):
        from petsc4py import PETSc

        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def F(self, x, b):
        """Assemble residual vector."""
        from petsc4py import PETSc

        from dolfinx.fem.petsc import apply_lifting, assemble_vector, set_bc

        with b.localForm() as b_local:
            b_local.set(0.0)
        assemble_vector(b, self.L)
        # Store internal_forces before applying BCs
        self.internal_forces = b.copy()
        apply_lifting(b, [self.a], bcs=[self.bc], x0=[x], alpha=-1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, self.bc, x, -1.0)

    def J(self, x, A):
        """Assemble Jacobian matrix."""
        from dolfinx.fem.petsc import assemble_matrix

        A.zeroEntries()
        assemble_matrix(A, self.a, bcs=self.bc)
        A.assemble()

    def matrix(self):
        from dolfinx.fem.petsc import create_matrix

        return create_matrix(self.a)

    def vector(self):
        from dolfinx.fem.petsc import create_vector

        return create_vector(self.L)
    
    def get_reaction_forces(self, boundary_dofs):
        # Return the *negative* of the internal forces at the specified dofs
        if self.internal_forces is None:
            raise ValueError("Solve the problem first to compute internal forces.")

        b_array = fem.Function(V)
        b_array.x.array[:V.dofmap.index_map.size_local*V.dofmap.index_map_bs] = self.internal_forces.getArray()
        b_array.x.scatter_forward()
        return b_array.x.array[boundary_dofs]
    
from dolfinx.nls.petsc import NewtonSolver
problem_u = NonlinearPDEProblem_ig(a_uv, Jac, Jac_imgrad, u, bcs)
def update(solver, dx, x):
    x.axpy(-1, dx)

solver = NewtonSolver(MPI.COMM_WORLD, problem_u)

solver.setF(problem_u.F, problem_u.vector())
solver.setJ(problem_u.J, problem_u.matrix())
solver.set_form(problem_u.form)
solver.set_update(update)
solver.error_on_nonconvergence = False
solver.convergence_criterion = "incremental"

solver.atol = 1.0e-8
solver.rtol = 1.0e-8

# The costomization of the Krylov solver should specified by the user regarding the problem size, computational resources, etc.

ksp1 = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp1.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}ksp_rtol"] = 1.0e-9
opts[f"{option_prefix}ksp_atol"] = 1.0e-9
opts[f"{option_prefix}ksp_converged_reason"] = None

opts[f"{option_prefix}pc_type"] = "lu"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
opts[f"{option_prefix}ksp_error_if_not_converged"] = False
opts[f"{option_prefix}matptap_via"] = "scalable"
opts[f"{option_prefix}options_left"] = None
ksp1.setFromOptions()


#ns = build_nullspace(V)
#problem_u.A.setNearNullSpace(ns)
#problem_u.A.setOption(PETSc.Mat.Option.SPD, True)  # type: ignore



# Create nonlinear problem
problem_z = NonlinearPDEProblem(a_z, Jac_z, z, bcs_z)

solver_z = NewtonSolver(MPI.COMM_WORLD, problem_z)
solver_z.setF(problem_z.F, problem_z.vector())
solver_z.setJ(problem_z.J, problem_z.matrix())
solver_z.set_form(problem_z.form)
solver_z.set_update(update)
solver_z.error_on_nonconvergence = False

solver_z.atol = 1.0e-7
solver_z.rtol = 1.0e-7

ksp2 = solver_z.krylov_solver
opts = PETSc.Options()
option_prefix = ksp2.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}ksp_rtol"] = 1.0e-8
opts[f"{option_prefix}ksp_atol"] = 1.0e-8
opts[f"{option_prefix}ksp_converged_reason"] = None
opts[f"{option_prefix}pc_type"] = "lu"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
opts[f"{option_prefix}ksp_error_if_not_converged"] = False
opts[f"{option_prefix}matptap_via"] = "scalable"
opts[f"{option_prefix}options_left"] = None
ksp2.setFromOptions()



ldot = 5*10**(-1)
maxdisp = 10    # Maximum displacement in mm

# time-stepping parameters
T = maxdisp / (ldot/2)
Totalsteps=500


startstepsize=T/Totalsteps

min_timestep = startstepsize*10**(-6)     # minimum possible time steps
max_timestep = startstepsize*10**(1)     # maximum possible time steps

stepsize=startstepsize
t=stepsize
step=1

nostagiter_g = 25 # stag for u and z

printsteps = 5  # for visualization


Omega = 1e9
problem_u.Omega = Omega
counter_for_omega = 0

norm_delu_tol = 1e-5
norm_delz_tol = 1e-6


while t-stepsize < T:

    if rank==0:
        print('Step= %d' %step, 't= %f' %t, 'Stepsize= %e' %stepsize)
        print('Omega= %e' %Omega)
        print('displacement_delta= %f' %(t/T*maxdisp))



    bcr0.g.value[...] = ScalarType(t/T*maxdisp)
    bcl0.g.value[...] = ScalarType(-t/T*maxdisp)
    stag_iter = 1
    norm_delu = 1
    norm_delz = 1
    while stag_iter<nostagiter_g and (norm_delu > norm_delu_tol or norm_delz > norm_delz_tol):
        start_time=time.time()


        ##############################################################
        # PDE for u
        ##############################################################
        if rank==0:
            print(f"solving for u in u-z staggered number {stag_iter}")
        n_u, converged_u = solver.solve(u)
        if rank==0:
            print(f"Newton iterations: {n_u}, Converged?: {converged_u}")
        if converged_u == False:
            break
        ##############################################################
        # PDE for z
        ##############################################################
        if rank==0:
            print(f"solving for z in u-z staggered number {stag_iter}:")
        n_z, converged_z = solver_z.solve(z)
        if rank==0:
            print(f"Newton iterations: {n_z}, Converged?: {converged_z}")
        z.x.scatter_forward()
        if converged_z == False:
            break
        ##############################################################

        zmin = domain.comm.allreduce(np.min(z.x.array), op=MPI.MIN)


        if rank==0:
            print(zmin)

        if rank==0:
            print("--- %s seconds ---" % (time.time() - start_time))

        ###############################################################
        #Residual check for stag loop
        ###############################################################


        error_u.x.array[:] = u.x.array - u_prev_iter.x.array
        norm_delu = norm_L2(comm, error_u)/norm_L2(comm, u_prev_iter)

        error_z.x.array[:] = z.x.array - z_prev_iter.x.array
        norm_delz = norm_L2(comm, error_z)/norm_L2(comm, z_prev_iter)

        u_prev_iter.x.array[:] = u.x.array
        z_prev_iter.x.array[:] = z.x.array

        if rank==0:
            print("Staggered Iteration after the whole u-z for u: {}, Norm = {}".format(stag_iter, norm_delu))
            print("Staggered Iteration after the whole u-z for z: {}, Norm = {}".format(stag_iter, norm_delz))



        stag_iter+=1

    ########### Post-processing ##############

    # time stepping variable
    re_calculate = False
    if np.isnan(norm_delu) or np.isnan(norm_delz) or converged_u == False or converged_z == False:
        t -= stepsize
        stepsize *= 0.75
        stepsize = max(stepsize, min_timestep)
        if not converged_u:
            Omega = ScalarType(max(Omega/10, 1e0))  # Reduce Omega if u did not converge
            problem_u.Omega = Omega
        re_calculate = True
        counter_for_omega = 0
        if rank == 0:
            print("Not converging. Reducing stepsize to %e and restarting iteration." % stepsize)

    elif norm_delu < 1e-3 and norm_delz < 1e-3 and stag_iter < 4:
        stepsize *= 1.1
        stepsize = min(stepsize, max_timestep)
        problem_u.Omega = Omega
        if rank == 0 and stepsize != max_timestep:
            print("norm_delu and norm_delz are small. Increasing stepsize to %e for the next iteration." % stepsize)




    # Calculate Reaction

    if not re_calculate:
        Fx = domain.comm.allreduce(np.sum(problem_u.get_reaction_forces(dofs_right0)), op=MPI.SUM)
        z_x = evaluate_function(z, (thickness/2,B-dcrack-1*eps,-H/2))[0]


        if rank==0:
            print(Fx)
            print(z_x)
            with open(f"trousers-elastic-phase-field_{ldot}.txt", 'a') as rfile:
                rfile.write("%s %s %s %s %s\n" % (str(t),str(t*maxdisp*2/T+distance), str(zmin), str(z_x), str(Fx)))

    if step % printsteps==0 and not re_calculate:
        uplot.x.array[:] = (local_project(u, Vplot)).x.array

        vtk = io.VTKFile(domain.comm, "Paraview_Files/trousers_paraview/trousers_" + str(step) + ".pvd", "w")
        vtk.write_function([uplot, z], t)
        vtk.close()

    if stepsize < 10**(1)*min_timestep:
        break


    # time stepping
    if not re_calculate:
        un.x.array[:] = u.x.array
        zn.x.array[:] = z.x.array
        step += 1
        t += stepsize
    else:
        if rank == 0:
            print("Everything back to the previous step values")
        u.x.array[:] = un.x.array
        z.x.array[:] = zn.x.array       
        t += stepsize