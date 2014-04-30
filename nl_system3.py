"""
    Solve the system of nonlinear system 

    -u1" + u2' + u3  +                        u1*u3 = f1
    -u1' - u2" + u3  +                   - 16*u2*u3 = f2
     u1  + u2  - u3" +  u1*u3 - 16*u2*u3 +    u3^2  = 0 


    with the boundary conditions 

    u1(0)=0, u1(1)=1
    u2(0)=1, u2(1)=0
    u3(0)=0, u3(1)=0

"""

from dolfin import *

class NonlinearSystem(NonlinearProblem):
    def __init__(self, a, L):
        NonlinearProblem.__init__(self)
        self.L = L
        self.a = a
        self.reset_sparsity = True
    def F(self, b, x):
        assemble(self.L, tensor=b)
    def J(self, A, x):
        assemble(self.a, tensor=A, reset_sparsity=self.reset_sparsity)
        self.reset_sparsity = False


# Number of elements
nel = 30

# Polynomial order
p = 2

# Create mesh and function space
mesh = IntervalMesh(nel,0,1)

V = FunctionSpace(mesh,"CG",p)

# Product space 
V3 = MixedFunctionSpace([V,V,V])

u = Function(V3)
du = TrialFunction(V3)
v = TestFunction(V3)

# Define boundary values 
bval = lambda a,b: Expression("({0}*(1-x[0])+{1}*x[0])/{2}".format(a,b,1))

def u0_boundary(x,on_boundary):
    return on_boundary

# Dirichlet boundary conditions
bc1 = DirichletBC(V3.sub(0),bval(-1,1),u0_boundary)
bc2 = DirichletBC(V3.sub(1),bval(0,-1),u0_boundary)
bc3 = DirichletBC(V3.sub(2),bval(1,0),u0_boundary)

bcs = [bc1,bc2,bc3]

# Define variational problem
u1,u2,u3 = split(u)
v1,v2,v3 = split(v)

f1 = Expression("pow(x[0]-1,2)-2")
f2 = Expression("pow(x[0],2)-2")
f3 = Expression("0*x[0]")


F = inner(grad(v1),grad(u1))*dx + \
    v1*u2.dx(0)*dx + \
    v1*u3*dx - \
    v2*u1.dx(0)*dx + \
    inner(grad(v2),grad(u2))*dx + \
    v2*u3*dx + \
    v3*u1*dx + \
    v3*u2*dx + \
    (v1*u1*u3-16*v2*u2*u3+v3*(u1-16*u2+u3)*u3)*dx + \
    v3*u3*dx + inner(grad(v3),grad(u3))*dx - f1*v1*dx - f2*v2*dx

dF = derivative(F, u, du)
   
 

problem = NonlinearVariationalProblem(F,u,bcs,J=dF)
pdesys_newton = NonlinearVariationalSolver(problem)
pdesys_newton.solve()
u1,u2,u3 = u.split()
plot(u1,title="u_1")
plot(u2,title="u_2")
plot(u3,title="u_3")
interactive()
