""" 
    bvp1.py solves the 1D Poisson equation using FEniCS 
    Copyright (C) 2013  Greg von Winckel

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

   Solve the 1D Poisson equation -u''=-1 with u(0)=0, u'(1)=1
   The exact solution is x^2/2
"""

from dolfin import *

# Number of elements
nel = 20

# Left end point 
xmin = 0

# Right end point 
xmax = 1

# Polynomial order of trial/test functions
p = 2

# Create mesh and function space
mesh = IntervalMesh(nel,xmin,xmax)

# Define function space for this mesh using Continuous Galerkin
# (Lagrange) functions of order p on each element
V = FunctionSpace(mesh,"CG",p)

# Define boundary boundary values
u0 = Expression("x[0]")

# This imposes a Dirichlet condition at the point x=0
def Dirichlet_boundary(x,on_boundary):
    tol = 1e-14
    return on_boundary and abs(x[0])<tol

# Enforce u=u0 at x=0
bc = DirichletBC(V,u0,Dirichlet_boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-1)
g = Constant(1) 
a = inner(grad(u),grad(v))*dx
L = f*v*dx+g*v*ds

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# plot solution
plot(u)

# Dump solution to file in VTK format
file = File("bvp1.pvd")
file << u

# hold plot
interactive()

