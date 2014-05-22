import numpy as np
import matplotlib.pyplot as plt
import dolfin as df
import ufl


if __name__ == '__main__':

    # Number of elements
    nel = 200

    # Local approximation order
    p = 2

    # Strength of nonlinearity 
    a = df.Constant(5.0)

    # Construct mesh on [0,1]
    mesh = df.IntervalMesh(nel,0,1)

    # Identify boundaries
    tol = 1E-14

    def left_boundary(x, on_boundary):
        return on_boundary and abs(x[0]) < tol

    def right_boundary(x, on_boundary):
        return on_boundary and abs(x[0]-1) < tol

    # Define function space
    V = df.FunctionSpace(mesh,"Lagrange",p)
    V2 = df.MixedFunctionSpace([V,V])

    u = df.Function(V2)
    du = df.TrialFunction(V2)
    v = df.TestFunction(V2)

    u1,u2 = df.split(u)
    v1,v2 = df.split(v)


    # Initial Guesses 
    u1_i = df.Expression("x[0]")
    u2_i = df.Expression("1-x[0]")

    # Impose boundary conditions for the solution to the nonlinear system
    bc = [df.DirichletBC(V2.sub(0),df.Constant(0.0),left_boundary),
          df.DirichletBC(V2.sub(0),df.Constant(1.0),right_boundary),
          df.DirichletBC(V2.sub(1),df.Constant(1.0),left_boundary),
          df.DirichletBC(V2.sub(1),df.Constant(0.0),right_boundary)]

    # Make homogeneous version of BCs for the update
    bch = df.homogenize(bc)  

    # Set the initial guess 
    u.interpolate(df.Expression(("u1_i","u2_i"),u1_i=u1_i,u2_i=u2_i))

    # Allocate Newton update function
    u_inc = df.Function(V2)

    # Extract grid points
    xg = mesh.coordinates()

    # Nonlinear coefficient 1
    q1 = ufl.operators.exp(a*u1)

    # Nonlinear coefficient 2
    q2 = ufl.operators.exp(a*u2)


    for k in range(10):

        # Two parts of the weak form
        F1 = (u1.dx(0)*v1.dx(0)+a*u1.dx(0)*q2*v1)*df.dx
        F2 = (u2.dx(0)*v2.dx(0)-a*q1*u2.dx(0)*v2)*df.dx

        F = F1 + F2

        # Compute Jacobian
	dF = df.derivative(F,u,du)

        # Assemble matrix and load vector
	A,b = df.assemble_system(dF,-F,bch)

        # Compute Newton update
	df.solve(A,u_inc.vector(),b)

        # Update solution
	u.vector()[:] += u_inc.vector()

        
    # Evaluate solution
    ug = u.compute_vertex_values()

    # partition into u1 and u2
    U = np.reshape(ug,(2,nel+1))

    plt.plot(xg,U[0,:])
    plt.plot(xg,U[1,:])
    plt.show()
