from mpi4py import MPI
from dolfinx import mesh, fem, io, plot, nls, log, geometry, la
import basix
import dolfinx.fem.petsc
import ufl
import numpy as np
from petsc4py.PETSc import ScalarType
from petsc4py import PETSc
import time
import os
log.set_log_level(log.LogLevel.WARNING)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
comm_rank = rank


# Material properties

E, nu = ScalarType(70000), ScalarType(0.22)	                                      #Young's modulus and Poisson's ratio
mu, lmbda, kappa = E/(2*(1 + nu)), E*nu/((1 + nu)*(1 - 2*nu)), E/(3*(1 - 2*nu))
Gc= ScalarType(0.010)	                                                          #Critical energy release rate
sts, scs= ScalarType(40), ScalarType(1000)	                                      #Tensile strength and compressive strength
shs = ScalarType(27.8)
Wts = sts**2/(2*E)
Whs = shs**2/(2*kappa)



#The regularization length
eps = 0.08
h = 0.015
# The delta parameter
delta = (1+3*h/(8*eps))**(-2) * ((sts + (1+2*np.sqrt(3))*shs)/((8+3*np.sqrt(3))*shs)) * 3*Gc/(16*Wts*eps) + (1+3*h/(8*eps))**(-1) * (2/5)


#Geometry

L = 5
Rad = 3
thickness = 0.15
R_in = Rad - thickness 




###########################################

# Importing the mesh from GMSH
partitioner = dolfinx.cpp.mesh.create_cell_partitioner(dolfinx.mesh.GhostMode.shared_facet)
mesh_data = io.gmshio.read_from_msh("tube.msh", MPI.COMM_WORLD, gdim=3, partitioner=partitioner)
domain = mesh_data[0]





# Defining the function spaces
V = fem.functionspace(domain, ("CG", 1, (domain.geometry.dim,)))                  #Function space for u
Y = fem.functionspace(domain, ("CG", 1))                                          #Function space for z


# Define boundary conditions
def bottom(x):
    return np.isclose(x[2], 0)

def top(x):
    return np.isclose(x[2], L)

fdim = domain.topology.dim -1

bottom_facets = mesh.locate_entities_boundary(domain, fdim, bottom)
top_facets = mesh.locate_entities_boundary(domain, fdim, top)

dofs_bottom0 = fem.locate_dofs_topological(V.sub(0), fdim, bottom_facets)
dofs_bottom1 = fem.locate_dofs_topological(V.sub(1), fdim, bottom_facets)
dofs_bottom2 = fem.locate_dofs_topological(V.sub(2), fdim, bottom_facets)

dofs_top = fem.locate_dofs_topological(V, fdim, top_facets)

dofs_top0 = fem.locate_dofs_topological(V.sub(0), fdim, top_facets)
dofs_top1 = fem.locate_dofs_topological(V.sub(1), fdim, top_facets)
dofs_top2 = fem.locate_dofs_topological(V.sub(2), fdim, top_facets)

bcb0 = fem.dirichletbc(ScalarType(0), dofs_bottom0, V.sub(0))
bcb1 = fem.dirichletbc(ScalarType(0), dofs_bottom1, V.sub(1))
bcb2 = fem.dirichletbc(ScalarType(0), dofs_bottom2, V.sub(2))


disp_top = fem.Function(V)
bct = fem.dirichletbc(disp_top, dofs_top)

bcs = [bcb0, bcb1, bcb2, bct]

xm = ufl.SpatialCoordinate(domain)
n = ufl.FacetNormal(domain)
tau = fem.Constant(domain, 0.0)
T_mat = ufl.as_matrix([[ufl.cos(tau), -ufl.sin(tau), 0], [ufl.sin(tau), ufl.cos(tau), 0], [0, 0, 1]])
rotational_disp = ufl.dot(T_mat, xm) - xm


expr = fem.Expression(rotational_disp, V.element.interpolation_points, comm)

bcs_z = []



marked_facets = np.hstack([bottom_facets, top_facets])
marked_values = np.hstack([np.full_like(bottom_facets, 1),
                           np.full_like(top_facets, 2)])
sorted_facets = np.argsort(marked_facets)
facet_tag = mesh.meshtags(domain, domain.topology.dim -1,
                          marked_facets[sorted_facets],
                          marked_values[sorted_facets])

metadata = {"quadrature_degree": 4}
ds = ufl.Measure('ds', domain=domain,
                 subdomain_data=facet_tag, metadata=metadata)
dS = ufl.Measure("dS", domain=domain, metadata=metadata)
dx = ufl.Measure("dx", domain=domain, metadata=metadata)

# Define functions
du = ufl.TrialFunction(V)                                                         # Incremental displacement
v  = ufl.TestFunction(V)                                                          # Test function for u
u  = fem.Function(V, name="displacement")                                         # Displacement from previous iteration
u_inc = fem.Function(V)
dz = ufl.TrialFunction(Y)                                                         # Incremental phase field
y  = ufl.TestFunction(Y)                                                          # Test function for z
z  = fem.Function(Y, name="phasefield")                                           # Phase field from previous iteration
z_inc = fem.Function(Y)
d = len(u)


# Initialize the functions
u.x.array[:] = 0.
z.x.array[:] = 1.


u_prev = fem.Function(V)
u_prev.x.array[:] = u.x.array
z_prev = fem.Function(Y)
z_prev.x.array[:] = z.x.array



un = fem.Function(V)
un.x.array[:] = u.x.array
zn = fem.Function(Y)
zn.x.array[:] = z.x.array


error_u = fem.Function(V)
error_u.x.array[:] = 0
error_z = fem.Function(Y)
error_z.x.array[:] = 0


Vplot = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))        #Function space for plotting u
uplot = fem.Function(Vplot, name="displacement")


def norm_L2(comm, v):
    """Compute the L2(O)-norm of v"""
    return np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(v, v) * dx)), op=MPI.SUM))


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



# Stored energy, strain and stress functions in linear isotropic elasticity

def energy(v):
    return mu*(ufl.inner(epsilon(v),epsilon(v))) + 0.5*(lmbda)*(ufl.tr(epsilon(v)))**2

def epsilon(v):
	return ufl.sym(ufl.grad(v))

def sigma(v):
    return 2.0*mu*epsilon(v) + (lmbda)*ufl.tr(epsilon(v))*ufl.Identity(3)


def epsilonD(v):
    return epsilon(v) - (1/3)*ufl.tr(epsilon(v))*ufl.Identity(3)

def sigmaD(v):
    return 2.0*mu*epsilonD(v)
 
def sigmavm(v):
    return ufl.sqrt(1/2*(ufl.inner(sigmaD(v),sigmaD(v))))

eta = 0.0
# Stored energy density
psi1 = (z**2+eta)*(energy(u))
psi11 = energy(u)
# Total potential energy
Pi = psi1*dx
# Compute first variation of Pi (directional derivative about u in the direction of v)
R = ufl.derivative(Pi, u, v)
# Compute Jacobian of R
Jac = ufl.derivative(R, u, du)


I1 = (z**2)*ufl.tr(sigma(u))
SQJ2 = (z**2)*sigmavm(u)

alpha1 = (delta*Gc)/(shs*8*eps) - (2*Whs)/(3*shs)
alpha2 = (3**0.5*(3*shs - sts)*delta*Gc)/(shs*sts*8*eps) + (2*Whs)/(3**0.5*shs) - (2*3**0.5*Wts)/(sts)

ce= alpha2*SQJ2 + alpha1*I1 - z*(1-ufl.sqrt(I1**2)/I1)*psi11

#Balance of configurational forces PDE
pen=1000*(3*Gc/8/eps)*ufl.conditional(ufl.lt(delta,1),1, delta)
Wv=pen/2*((abs(z)-z)**2 + (abs(1-z) - (1-z))**2 )*dx
Wv2=ufl.conditional(ufl.le(z, 0.95), 1, 0)*100*pen/2*((1/4)*(abs(zn-z)-(zn-z))**2)*dx

R_z = y*2*z*(psi11)*dx + y*(ce)*dx + 3*delta*Gc/8*(-y/eps + 2*eps*ufl.inner(ufl.grad(z),ufl.grad(y)))*dx\
      + ufl.derivative(Wv,z,y)+ ufl.derivative(Wv2,z,y)

# Compute Jacobian of R_z
Jac_z = ufl.derivative(R_z, z, dz)


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
    
from dolfinx.nls.petsc import NewtonSolver
problem_u = NonlinearPDEProblem(R, Jac, u, bcs)
def update(solver, dx, x):
    x.axpy(-1, dx)

solver = NewtonSolver(MPI.COMM_WORLD, problem_u)
solver.setF(problem_u.F, problem_u.vector())
solver.setJ(problem_u.J, problem_u.matrix())
solver.set_form(problem_u.form)
solver.set_update(update)
solver.error_on_nonconvergence = False

solver.atol = 1.0e-6
solver.rtol = 1.0e-7

ksp1 = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp1.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}ksp_rtol"] = 1.0e-8
opts[f"{option_prefix}ksp_atol"] = 1.0e-8
opts[f"{option_prefix}ksp_converged_reason"] = None

opts[f"{option_prefix}pc_type"] = "lu"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
opts[f"{option_prefix}ksp_error_if_not_converged"] = False
opts[f"{option_prefix}matptap_via"] = "scalable"
opts[f"{option_prefix}options_left"] = None
ksp1.setFromOptions()




# Create nonlinear problem
problem_z = NonlinearPDEProblem(R_z, Jac_z, z, bcs_z)

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




# time-stepping parameters
ldot = 5*10**(-1)
tau_max = L*0.000154704*5

T = tau_max / (ldot)


Totalsteps=300
startstepsize=T/Totalsteps
stepsize=startstepsize
t=stepsize
step=1
printsteps = 10


while t-stepsize < T:

    if comm_rank==0:
        print('Step= %d' %step, 't= %f' %t, 'Stepsize= %e' %stepsize)

    # Update the displacement at the top boundary

    tau.value = tau_max*t/T
    disp_top.interpolate(expr)

    stag_iter = 1
    norm_delu = 1
    norm_delz = 1
    while stag_iter<100 and (norm_delu > 1e-6 or norm_delz > 1e-6):
        start_time=time.time()
        ##############################################################
        # PDE for u
        ##############################################################
        if rank==0:
            print(f"solving for u in u-z staggered number {stag_iter}:")
        n_u, converged_u = solver.solve(u)
        if rank==0:
            print(f"Newton iterations: {n_u}, Converged?: {converged_u}")
        u.x.scatter_forward()
        ##############################################################
        # PDE for z
        ##############################################################
        if rank==0:
            print(f"solving for z in u-z staggered number {stag_iter}:")
        n_z, converged_z = solver_z.solve(z)
        if rank==0:
            print(f"Newton iterations: {n_z}, Converged?: {converged_z}")
        z.x.scatter_forward()
        ##############################################################

        zmin = domain.comm.allreduce(np.min(z.x.array), op=MPI.MIN)


        if rank==0:
            print(zmin)

        if rank==0:
            print("--- %s seconds ---" % (time.time() - start_time))

        ###############################################################
        # Residual check for stag loop
        ###############################################################


        error_u.x.array[:] = u.x.array - u_prev.x.array
        norm_delu = norm_L2(comm, error_u)/norm_L2(comm, u_prev)

        error_z.x.array[:] = z.x.array - z_prev.x.array
        norm_delz = norm_L2(comm, error_z)/norm_L2(comm, z_prev)

        if rank==0:
            print("Staggered Iteration after the whole u-z for u: {}, Norm = {}".format(stag_iter, norm_delu))
            print("Staggered Iteration after the whole u-z for z: {}, Norm = {}".format(stag_iter, norm_delz))

        u_prev.x.array[:] = u.x.array
        z_prev.x.array[:] = z.x.array
        stag_iter+=1


    ########### Post-processing ##############


    un.x.array[:] = u.x.array
    zn.x.array[:] = z.x.array

    # Calculate Reaction

    Fx1 = domain.comm.allreduce(np.sum(problem_u.get_reaction_forces(dofs_top0)), op=MPI.SUM)
    Fx2 = domain.comm.allreduce(np.sum(problem_u.get_reaction_forces(dofs_top1)), op=MPI.SUM)
    Fx = np.sqrt(Fx1**2 + Fx2**2)

    M1_expr = ufl.cross(xm, ufl.dot(sigma(u), n))*z**2

    M1 = domain.comm.allreduce(fem.assemble_scalar(fem.form(M1_expr[2]*ds(1))), op=MPI.SUM)

    z_x = evaluate_function(z, (Rad, 0, L/2))[0]


    if rank==0:
        print(Fx)
        print(z_x)
        with open('Elastic_phasefield_torsion.txt', 'a') as rfile:
            rfile.write("%s %s %s %s %s\n" % (str(t), str(t/T*tau_max), str(zmin), str(z_x), str(-M1)))

    if zmin < 0:
        printsteps = 1

    if step % printsteps==0:
        uplot.x.array[:] = (local_project(u, Vplot)).x.array
        vtk = io.VTKFile(domain.comm, "paraview/3D_elast" + str(step) + ".pvd", "w")
        vtk.write_function([uplot, z], t)
        vtk.close()
        

    # time stepping
    step+=1
    t+=stepsize
