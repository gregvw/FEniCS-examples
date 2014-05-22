FEniCS-examples
===============

Example Python codes for FEniCS 

---

`bvp1.py` is the simplest example and solves the linear
2-point boundary value problem 

    -u"(x)=-1 

with the boundary conditions

    u(0) = 0, u'(1) = 1

---

`bvp2.py` solves a system of two coupled boundary value problems

    -u1" +  u2  = f1
     u1  -  u2" = f2

with the boundary conditions

    u1(0)=0, u1(1)=1
    u2(0)=1, u2(1)=0

---

`nl_system3.py` solves a system of 3 nonlinear boundary value problems

    -u1" + u2' + u3  +                        u1*u3 = f1
    -u1' - u2" + u3  +                   - 16*u2*u3 = f2
     u1  + u2  - u3" +  u1*u3 - 16*u2*u3 +    u3^2  = 0 

with the boundary conditions 

    u1(0) = 0,  u1(1) = 1
    u2(0) = 1,  u2(1) = 0
    u3(0) = 0,  u3(1) = 0

---

`adv_diff.py` solves the advection diffusion equation

    u_t + a*u_x = d*u_xx

with a given initial condition and homogeneous Neumann conditons

    u_x(-1) = u_x(+1) = 0

---

`newton.py' solves the nonlinear boundary value problem

    (exp(a*u)*u')' = 0
    u(0) = 0, u(1) = 1

Using a manual Newton's method with continuation/homotopy

---

`newton_sys2.py' solves the two coupled nonlinear boundary value problems

    -u1" + a*exp(u2)*u1 = 0
    -u2" + a*exp(u1)*u2 = 0

    u1(0) = 0,  u1(1) = 1
    u2(1) = 1,  u2(0) = 0    

---

`finite_well.py` computes eigenvalues and eigenfunctions for 
a finite potential quantum well. 






