"""
    Solve the stationary viscous Burgers' equation using FEniCS
    -mu*u"+(u*u_x)=0
     u(-1)=1, u(+1)=1
"""

import dolfin as df

# Number of elements
nel = 30

# Left end point 
xmin = -1

# Right end point 
xmax = 1

# Polynomial order of trial/test functions
p = 2

# Create mesh and function space
mesh = df.IntervalMesh(nel,xmin,xmax)

# Define function space for this mesh using Continuous Galerkin
# (Lagrange) functions of order p on each element
V = df.FunctionSpace(mesh,"CG",p)


def u0_boundary(x,on_boundary):
    return on_boundary


# Define function for setting Dirichlet values
bval = lambda lv,rv: df.Expression("({0}*(1-x[0])+{1}*(1+x[0]))/2".format(lv,rv))

bcs = df.DirichletBC(V,bval(1,-1),u0_boundary)

dx = df.Measure("dx")

# Define variational problem
u = df.Function(V)
u_x = u.dx(0)

v = df.TestFunction(V)
v_x = v.dx(0)

mu = df.Constant(0.1)

F = (mu*u_x*v_x+v*u*u_x)*dx


df.solve(F == 0,u, bcs,solver_parameters={"newton_solver":
                                         {"relative_tolerance":1e-6}})


# plot solution
df.plot(u,title="Velocity")


# hold plot
df.interactive()
