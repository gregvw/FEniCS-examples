"""

   quantum_dot.py numerically computes the ground state energy and 
   eigenfunction of a hemispherical quantum dot using FEniCS

"""

import dolfin as df
import numpy as np

if __name__ == '__main__': 
    
    # Computational domain radius
    r_dom = 2

    # Minimum computational domain height
    z_dom_min = -1

    # Maximum computational domain height 
    z_dom_max = 2

    # Number of cells in the radial direction
    nr = 100
 
    # Number of cells in the radial direction
    nz = 100   

    mesh = df.RectangleMesh(0,z_dom_min,r_dom,z_dom_max,nr,nz)
    
    class Top(df.SubDomain):
        def inside(self, x, on_boundary):
            return df.near(x[1],z_dom_max)

    class Bottom(df.SubDomain):
        def inside(self, x, on_boundary):
            return df.near(x[1],z_dom_min)

    class Outer(df.SubDomain):
        def inside(self, x, on_boundary):
            return df.near(x[0],r_dom)

    class Inner(df.SubDomain):
        def inside(self, x, on_boundary):
            return df.near(x[0],0)

    class QuantumDot(df.SubDomain):
        def inside(self,x,on_boundary):
            return df.between(x[0]**2+x[1]**2,(0,1)) and df.between(x[1],(0,1))   


    quantumDot = QuantumDot()

    domains = df.CellFunction("size_t",mesh)
    domains.set_all(0)
    quantumDot.mark(domains,1)

    inner = Inner()
    outer = Outer()
    top = Top()
    bottom = Bottom()

    boundaries = df.FacetFunction("size_t",mesh)
    boundaries.set_all(0)

    top.mark(boundaries,1)
    outer.mark(boundaries,2)
    bottom.mark(boundaries,3)

    V = df.FunctionSpace(mesh,"CG",2)

    # Apply Homogeneous Dirichlet conditions on top, bottom, and outer wall of 
    # computational domain
    bcs = [df.DirichletBC(V,0.0,boundaries,1),
          df.DirichletBC(V,0.0,boundaries,2),
          df.DirichletBC(V,0.0,boundaries,3)]

    u = df.TrialFunction(V)
    v = df.TestFunction(V)     
 
    drdz = df.Measure("dx")[domains]
    r = df.Expression("x[0]")

    potential = df.Constant(100)

    u_r = u.dx(0)
    v_r = v.dx(0)
    u_z = u.dx(1)
    v_z = v.dx(1)

    # Time step size
    dt = df.Constant(0.01)

    # Initial guess of ground state is 1 inside dot, 0 outside dot
    psi0 = v*r*drdz(1)+df.Constant(0)*v*r*drdz(0)
    Psi0 = df.PETScVector()
    df.assemble(psi0,tensor=Psi0)


    # Hamiltonian and mass matrix forms
    h = (u_r*v_r+u_z*v_z)*r*(drdz(0)+drdz(1))+potential*r*u*v*r*drdz(0)
    m = (u*v*r)*(drdz(0)+drdz(1))
    
    A = df.PETScMatrix()
    df.assemble(m+dt*h,tensor=A)

    M = df.PETScMatrix()
    df.assemble(m,tensor=M)

    H = df.PETScMatrix()
    df.assemble(h,tensor=H)
 
    # Apply boundary conditions 
    for bc in bcs:
         bc.apply(A)
         bc.apply(Psi0)

    psi = df.Function(V)
    solver = df.PETScLUSolver(A)
    solver.parameters['symmetric'] = True

    solver.solve(psi.vector(),Psi0) 
 
    q = psi.vector()
    
    for k in range(30):
        Mq = M*q
        qHq = q.inner(H*q)
        qMq = q.inner(Mq)

        # Rayleigh quotient
        E = qHq/qMq
        print(E)
 
        q /= np.sqrt(qMq)

        solver.solve(q,Mq) 

    Mq = M*q
    q /= np.sqrt(q.inner(Mq))

    psi.vector()[:] = q
  
    df.plot(psi,title="Ground State")
    df.interactive() 
