"""
   Solve the nonlinear boundary value problem
  
   -(exp(a*u)*u')' = 0, u(0) = 0, u(1) = 1

   using Newton's method with continuuation (homotopy method) 

   To do this, we introduce the paramater 0<g<=1
   and solve the problem

   -([g*exp(a*u)+(1-g)]*u')' = 0 

   where g moves gradually from 0 to 1

   For the example with a=10, the standard Newton method does not 
   converge

"""


import numpy as np
import dolfin as df
import ufl
import matplotlib.pyplot as plt
            
if __name__ == '__main__':

    # Number of elements
    nel = 200

    # Local approximation order
    p = 2

    # Strength of nonlinearity 
    a = 10    

    # Construct mesh on [0,1]
    mesh = df.IntervalMesh(nel,0,1)

    # Define function space
    V = df.FunctionSpace(mesh,"Lagrange",1)

    u = df.Function(V)
    v = df.TestFunction(V)

    du = df.TrialFunction(V)

    # Identify boundaries
    tol = 1E-14

    def left_boundary(x, on_boundary):
        return on_boundary and abs(x[0]) < tol

    def right_boundary(x, on_boundary):
        return on_boundary and abs(x[0]-1) < tol

    # Define nonlinear term with continuation parameter
    def q(u,a,g):
        g1 = df.Constant(g)
        g2 = df.Constant(1-g)
        return g1*ufl.operators.exp(df.Constant(a)*u)+g2

    # Define boundary conditions for the solution
    Gamma_0 = df.DirichletBC(V, df.Constant(0.0), left_boundary)
    Gamma_1 = df.DirichletBC(V, df.Constant(1.0), right_boundary)
    bc = [Gamma_0, Gamma_1]

    # Make homogeneous equivalent of boundary conditions for update
    bch = df.homogenize(bc)

    # Define function for storing the Newton updates 
    u_inc = df.Function(V)

    # Initial guess of a function that satisfies the boundary conditions
    ui = df.Expression('x[0]')

    # Evaluate u using the initial data
    u.interpolate(ui)

    # Extract the mesh nodes
    xg = mesh.coordinates()

    # Set number of continuuation steps
    Nc = 20
    steps = [(float(k)/(Nc-1))**3 for k in range(Nc)]
    for s in steps:

        # Construct form and its Jacobian
        F = u.dx(0)*v.dx(0)*q(u,a,s)*df.dx
        dF = df.derivative(F,u,du)

        # Assemble the system for the Newton update with boundary conditions
        A,b = df.assemble_system(dF,-F,bch)
 
        # Solve for update
        df.solve(A,u_inc.vector(),b)

        # update solution
        u.vector()[:] += u_inc.vector()

        # Extract values 
        ug = u.compute_vertex_values()

        # Display iterate
        plt.plot(xg,ug)

    str_bvp = r'$\frac{d}{dx}\left[e^{10u}\frac{du}{dx}\right]=0$'
    str_bc = r'$u(0)=0,\; u(1)=1$'

    plt.title('Iterated Solution',fontsize=20)
    plt.text(0.6,0.4,str_bvp,fontsize=26)
    plt.text(0.6,0.2,str_bc,fontsize=18)
    plt.xlabel('x',fontsize=18)
    plt.ylabel('u(x)',fontsize=18)
    plt.show()
