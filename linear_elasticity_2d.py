from dolfinx import mesh, fem, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
import numpy as np

# Lamé parameters
mu = 1.0
lam = 1.25

# structured mesh made of quadrilaterals.
# Remark: to use a mesh with curved boundaries in FEniCSx, one needs to create it with Gmsh and
# then import it with the function "gmshio.read_from_msh"
domain = mesh.create_unit_square(MPI.COMM_WORLD, 50, 50, mesh.CellType.quadrilateral)

# polynomial degree
poly_degree = 1

# Vector space defined on the mesh "domain", made of Lagrange polynomials of degree poly_degree.
# "(domain.geometry.dim,)" is used to specify that the dimension of V is the same as the
# one of the ambient space
V = fem.functionspace(domain, ("Lagrange", poly_degree, (domain.geometry.dim,)))

# outward unit normal (needed?)
n = ufl.FacetNormal(domain)


############################# Boundary conditions #############################


# setting boundaries with corresponding labels:
# 1: left side (x = 0). Traction imposed.
# 2: bottom half of the right side (x = 1 and 0 <= y <= 0.5). x-component of the displacement
#    fixed.
# 3: bottom side (y = 0). y-component of the displacement fixed.
# 4: top half of the right side (x = 1 and 0.5 <= y <= 1) and top side (y = 1). Traction-free.
boundaries = [
    (1, lambda x: np.isclose(x[0], 0)),
    (2, lambda x: np.logical_and(np.isclose(x[0], 1), x[1] <= 0.5)),
    (3, lambda x: np.isclose(x[1], 0)),
    (
        4,
        lambda x: np.logical_or(
            np.logical_and(np.isclose(x[0], 1), x[1] >= 0.5), np.isclose(x[1], 1)
        ),
    ),
]


# loop through boundaries and identify the facets for each bc
facet_indices, facet_markers = [], []
fdim = domain.topology.dim - 1  # dimension of boundary elements
for marker, locator in boundaries:
    facets = mesh.locate_entities(domain, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full_like(facets, marker))
facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tag = mesh.meshtags(
    domain, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets]
)

# plot bcs
domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
with io.XDMFFile(domain.comm, "facet_tags.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_meshtags(facet_tag, domain.geometry)


# new measure for boundary integration (Neumann bcs)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)


# extract boundary elements where x-component of the displacement is fixed
boundary_facets_x = mesh.locate_entities_boundary(domain, fdim, boundaries[1][1])

# extract boundary elements where y-component of the displacement is fixed
boundary_facets_y = mesh.locate_entities_boundary(domain, fdim, boundaries[2][1])

# subset of the faces where the x-component of the displacement is fixed
boundary_dofs_x = fem.locate_dofs_topological(V.sub(0), fdim, boundary_facets_x)

# subset of the faces where the y-component of the displacement is fixed
boundary_dofs_y = fem.locate_dofs_topological(V.sub(1), fdim, boundary_facets_y)


# enforcement of "carrelli-type" bcs
bcx = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V.sub(0))
bcy = fem.dirichletbc(default_scalar_type(0), boundary_dofs_y, V.sub(1))

# resulting Dirichlet bcs
dirichlet_bcs = [bcx, bcy]


# strain tensor
def epsilon(u):
    return ufl.sym(
        ufl.grad(u)
    )  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)


# stress tensor
def sigma(u):
    return lam * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)


# outward unit normal vector
n = ufl.FacetNormal(domain)

# vector for traction-free boundary: null traction
T0 = fem.Constant(domain, default_scalar_type((0.0, 0.0)))

# vector for traction-imposed boundary
T = fem.Constant(domain, default_scalar_type(0.2)) * n

# trial and test functions
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# body forces: null in this case
f = fem.Constant(domain, default_scalar_type((0, 0)))

# bilinear and linear forms.
# In this case both f and T0 are null, so the only contribution to the linear form is given by T.
# For the sake of generality, we keep all the terms.
a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx + ufl.inner(T0, v) * ds(4) + ufl.inner(T, v) * ds(1)

# solution of the linear problem
problem = LinearProblem(
    a, L, bcs=dirichlet_bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
)
uh = problem.solve()

# Save solution in XDMF format
with io.XDMFFile(domain.comm, "deformation.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    uh.name = "Deformation"
    xdmf.write_function(uh)
