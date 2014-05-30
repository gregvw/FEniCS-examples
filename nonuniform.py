"""

   Example of using a nonuniform grid for selectively resolving portions
   of the computational domain

   Distribute n grid points on the interval [0,1] so that the density of nodes
   is higher at a set of m locations xi[0],...,xi[m-1]

   Test the grid clustering method on the boundary value problem

   -(p(x)u')' = 1, u(0)=u(1)=0

   where p(x) = 1.001+cos(4*pi*x)

   so that u varies rapidly in neighborhoods of the points where p(x) is small

"""

import numpy as np
import matplotlib.pyplot as plt
import dolfin as df


def update(q,y,xj,g):
    nj = len(xi)

    # Nonnormalized mapping function
    phi_hat = lambda q: sum([0.5+np.arctan((q-xj[j])/g[j])/np.pi 
                             for j in range(nj)])

    # Scaling for normalization
    scale = 1/(phi_hat(1)-phi_hat(0))
  
    # Normalized mapping function
    phi = scale*(phi_hat(q)-phi_hat(0))

    # Derivative of normalized mapping function
    phi_x = scale*sum([g[j]/(np.pi*((q-xj[j])**2+g[j]**2)) 
                       for j in range(nj)])

    # Newton update 
    dx = (y-phi)/phi_x

    return dx

def compute_grid(n,xi,g,nsteps,csteps):

    # Uniform grid points    
    xu = np.linspace(0,1,n)

    # Clustered grid points (initial guess)
    x = np.linspace(0,1,n)

    t = [float(i+1)/csteps for i in range(csteps)]
    g0 = np.ones(len(xi))

    for l in range(csteps):
        gt = t[l]*g + (1-t[l])*g0        

        for k in range(nsteps):    
            dx = update(x,xu,xi,gt)
            x[1:-1] += dx[1:-1]

    return x


# Boundaries
class BoundaryPoint(df.SubDomain):
    def __init__(self,bpt):
        df.SubDomain.__init__(self)
        self.bpt = bpt
    def inside(self,x,on_boundary):
        return df.near(x[0],self.bpt)


if __name__ == '__main__':

    # Number of grid points
    n = 200

    # Create FEniCS mesh
    mesh = df.IntervalMesh(n-1,0,1)

    # Clustering locations
    xi = np.array((0.25,0.75))

    # Number of junctions
    ni = len(xi)
 
    # Set node clustering factors (smaller means more clustering)
    g = np.array((0.02,0.02))

    # Compute clustered grid
    x = compute_grid(n,xi,g,3,100)

    # Modify mesh to use clustered grid
    mesh.coordinates()[:] = np.reshape(x,(n,1))

    # Set up function space
    V = df.FunctionSpace(mesh,"CG",1)
    u = df.TrialFunction(V)
    v = df.TestFunction(V)

    ux = u.dx(0)
    vx = v.dx(0)
   
    # Set up homogeneous Dirichlet boundary conditions 
    left = BoundaryPoint(0)
    right = BoundaryPoint(1)
   
    boundaries = df.FacetFunction("size_t",mesh)
    left.mark(boundaries,0)
    right.mark(boundaries,1) 

    bc = [df.DirichletBC(V,df.Constant(0.0),left),
          df.DirichletBC(V,df.Constant(0.0),right)]

    # Variable coefficient
    p = df.Expression('1.001+cos(4*pi*x[0])')

    # Forms
    a = (vx*p*ux)*df.dx
    L = v*df.dx

    u = df.Function(V)
    df.solve(a==L,u,bc)

    # Extract nodal values 
    ug = u.compute_vertex_values() 

    fig = plt.figure(1,(10,4))
    ax = fig.add_subplot(111)
       
    ax.plot(x,ug,'.')

    ylim = ax.get_ylim()
    ax.vlines(xi,ylim[0],ylim[1],linestyle='--')
    ax.set_xlim(0,1)
    plt.show()


    

