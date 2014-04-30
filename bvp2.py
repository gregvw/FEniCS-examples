""" 
    bvp2.py solves two coupled linear boundary value problems using FEniCS 
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

    Solve the two coupled linear boundary value problems

    -u1" + u2 = (x-1)^2 - 2 
     u1 - u2" = x^2 - 2

    where u1(0) = 0  u1(1) = 1 
          u2(0) = 1  u2(1) = 0

    The exact solution is u1 = x^2, u2 = (1-x)^2

"""

from dolfin import *


# Number of elements
nel = 30

# Polynomial order
p = 2

# Create mesh and function space
mesh = IntervalMesh(nel,0,1)

V = FunctionSpace(mesh,"CG",p)

# Product space 
V2 = V*V

# Define boundary values 
right_bdry = Expression("x[0]")
left_bdry  = Expression("1-x[0]")

def u0_boundary(x,on_boundary):
    return on_boundary

# Dirichlet boundary conditions
bc1 = DirichletBC(V2.sub(0),right_bdry,u0_boundary)
bc2 = DirichletBC(V2.sub(1),left_bdry,u0_boundary)
bcs = [bc1,bc2]

# Define variational problem
(u1,u2) = TrialFunctions(V2)
(v1,v2) = TestFunctions(V2)

f1 = Expression("pow(x[0]-1,2)-2")
f2 = Expression("pow(x[0],2)-2")

a = inner(grad(u1),grad(v1))*dx + u2*v1*dx + \
    u1*v2*dx + inner(grad(u2),grad(v2))*dx

L = f1*v1*dx + f2*v2*dx
s 
u = Function(V2)
solve(a == L, u, bcs)

u1,u2 = u.split()
plot(u1,title="u_1")
plot(u2,title="u_2")
interactive()
