from dolfin import *

# Test for PETSc and SLEPc
if not has_linear_algebra_backend("PETSc"):
    print "DOLFIN has not been configured with PETSc. Exiting" 
    exit()

if not has_slepc():
    print "DOLFIN has not been configured with SLEPc. Exiting"
    exit()


# hbar^2/2m0 scaled so that all units are in eV and Angstroms
hb2m = 3.81 

# Thickness of the well (in Angstroms)
dwell = 80
 
# Thickness of the barriers 
dbar = 250

# Total thickess of structure
dtot = 2*dbar + dwell

# Potential in the well (in eV)
vwell = 0.0

# Potential in the barrier
vbar = 0.292

# Potential in the three layers
vpot = []
vpot.append(vbar)
vpot.append(vwell)
vpot.append(vbar)
Vpot = list(map(lambda k:Constant(vpot[k]),range(3)))

# Effective mass in the well (multipler for free rest mass)
mwell = 0.067

# Effective mass in the barrier
mbar = 0.096

# Stiffness matrix factor in the three layers
f = []
f.append(Constant(hb2m/mbar))
f.append(Constant(hb2m/mwell))
f.append(Constant(hb2m/mbar))


# Layers are an interval SubDomain 
class Layer(SubDomain):
    def __init__(self,ends):
        SubDomain.__init__(self)
        self.ends = ends
       
    def inside(self,x,on_boundary):
        return between(x[0],self.ends)

# Boundaries
class BoundaryPoint(SubDomain):
    def __init__(self,bpt):
        SubDomain.__init__(self)
        self.bpt = bpt
    def inside(self,x,on_boundary):
        return near(x[0],self.bpt)


# Initialize list of subdomain instances
layers = []
layers.append(Layer((0,dbar)))
layers.append(Layer((dbar,dbar+dwell)))
layers.append(Layer((dbar+dwell,dtot)))

# Initialize boundary instances
left = BoundaryPoint(0)
right = BoundaryPoint(dtot)


# Number of elements
nel = 80

# Order of Lagrange Trial/Test functions
p = 2

# Create mesh
mesh = IntervalMesh(nel,0,dtot)

# Initialize mesh function for interior domains
domains = CellFunction("size_t",mesh)
for k in range(3):
    layers[k].mark(domains,k)


# Initialize mesh function for boundary domains
boundaries = FacetFunction("size_t",mesh)
left.mark(boundaries,0)
right.mark(boundaries,1)

# Define function spaces and basis functions
V = FunctionSpace(mesh,"CG",3)
u = TrialFunction(V)
v = TestFunction(V)

# Impose homogeneous Dirichlet conditions at left and right boundaries
bcs = [DirichletBC(V,0.0,left),DirichletBC(V,0.0,right)]

# Define new measure associated with subdomains
dx = Measure("dx")[domains]

# Variational forms for the left (FEM Hamiltonian) and right (Mass) matrices
h = sum(map(lambda k:  (inner(f[k]*grad(u),grad(v))+Vpot[k]*u*v)*dx(k),range(3))) 

# Add all mass matrix terms
m = sum(map(lambda k: u*v*dx(k),range(3))) 

    
# Assemble matrices
H = PETScMatrix()
M = PETScMatrix()
assemble(h,tensor=H)
assemble(m,tensor=M)

# Create eigensolver for generalized EVP
eigensolver = SLEPcEigenSolver(H,M)
eigensolver.parameters["spectrum"]="smallest magnitude"
eigensolver.parameters["solver"]="lapack"

# Compute generalized eigenvalue decomposition
print "Computing eigenvalue decomposition. This may take a while."
eigensolver.solve()

# Print and display all bound states
r = 0
dex = 0

E = []
Psi = []

while r<vbar:

    # Extract next smallest eigenpair
    r, c, rx, cx = eigensolver.get_eigenpair(dex)
    u = Function(V)
    u.vector()[:] = rx
    
    E.append(r)
    Psi.append(u)
    dex += 1
    

boundstates = len(E)-1

for k in range(boundstates):
    print "E[" + str(k) + "]=" + str(E[k])
    plot(Psi[k], title="Psi[" + str(k) + "]")
    
interactive()












