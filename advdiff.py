import warnings
import numpy as np

def advdiff(v = None,K = None,c0 = None,L = None,dt = None,T = None,dxmethod = None,dtmethod = None,theta = None):

    # advdiff - solve the advection-diffusion equation in one dimension on
    #           the interval [0,L]. The advection-diffusion equation is
    #           given by:
     
    #                      2
    #      dc     dc      d c
    #      -- + v --  - K --- = 0
    #      dt     dx        2
    #                     dx
     
    #    An initial condition c(x,0)= c0(x) must be given such that
    #    c0(0) = c0(L) = 0, so that no boundary conditions need to be imposed
     
    #    Input parameters
    #    ----------------
     
    #    c=advdiff(v, K, c0, L, dt, T, dxmethod, dtmethod, theta)
     
    #    v        - velocity
    #    K        - diffusion coefficient
    #    c0       - row vector of initial condition
    #    L        - the length of the domain
    #    dt       - time step
    #    T        - simulation end time
    #    dxmethod - name of spatial discretization
    #    dtmethod - name of time integration
    #    theta    - implicitness parameter
     
    #    Output parameter
    #    ----------------
     
    #    The output parameter is c which is the numerical solution of the
    #    advection-diffusion equation. The parameter c is a two-dimensional
    #    array c[0:N,0:M] with N the number of time steps and M the number
    #    of grid cells. So, the number of grid points is therefore M + 1
    #    (including the boundary points).
     
    #    Initial condition
    #    -----------------
     
    #    The row vector c0 represents the initial value of c at uniformly
    #    spaced points in [0,L]. Hence, the dimension of c0 is c0[0:M].
     
    #    Spatial discretization
    #    ----------------------
     
    #    The allowed values of dxmethod are:
     
    #    'central' - second order central differences
    #
    #    'upwind'  - first order upwind scheme
    #
    #    'bdf'     - second order BDF scheme
    #
    #    'quick'   - second order QUICK scheme
    #
    #    'box'     - spatial component of the box scheme
    #                note: you also need to use the 'box' method for time integration
     
    #    Time integration
    #    ----------------
     
    #    The allowed values of dtmethod are:
    #
    #    'theta'        - theta-method
    #                     special cases:
    #                       theta=0.0 is explicit Euler
    #                       theta=0.5 is Crank-Nicolson
    #                       theta=1.0 is implicit Euler
    #
    #    'box'          - theta-box scheme
    #                     the value of theta is 0.5 to 1
    #                     the Preissmann scheme is a special case with theta=0.5
    #
    #    'leapfrog'     - Leapfrog scheme
    #                     note: parameter theta is not used
    #
    #    'Lax-Wendroff' - Lax-Wendroff scheme
    #                     note: parameter theta is not used
    #
    #    'Beam-Warming' - Beam-Warming scheme
    #                     note: parameter theta is not used
    #
    #   Note that the diffusion term is only supported by central differences
    #   combined with the theta-method

    # set K to zero for BDF, QUICK and BOX schemes, or Leapfrog, Lax-Wendroff and Beam-Warming time stepping methods 
    if (dxmethod != 'central' and dxmethod != 'upwind') or dtmethod != 'theta':
       if K != 0.:
          K = 0.
          warnings.warn('Diffusion coefficient has been set to zero')

    # set number of time steps 
    timesteps = int(T/dt)+1

    # initialize solution matrix (c[n,:] is the solution at timestep n)
    c = np.zeros((timesteps,len(c0)))

    # copy the initial condition into the first row of the solution matrix
    c[0,:] = c0
    
    # if Leapfrog then also write it to the second row
    if dtmethod == 'leapfrog': c[1,:] = c0
    
    # number of grid points
    points = len(c[0])

    # spatial grid size
    dx = L / (points - 1)

    #------------------------------------------------------------------------
    # Generate the spatial discretization template
    #------------------------------------------------------------------------
    
    # Calculate the templates for the first and second derivatives
    dcdx1 = dcdx(dx,1,v,dxmethod)
    dcdx2 = dcdx(dx,2,v,dxmethod)

    # dc/dt = -v(dc/dx) + K(d^2c/dx^2)
    template = - v * dcdx1 + K * dcdx2

    # Special case for Lax-Wendroff and Beam-Warming since these schemes produce artificial diffusion
    if dtmethod == 'Lax-Wendroff' or dtmethod == 'Beam-Warming':
       template = - v * dcdx1 + 0.5 * dt * v**2 * dcdx2
    
    # For a larger stencil, like BDF and QUICK, also a lower stencil of upwind to be applied to boundaries
    dcdx1 = dcdx(dx,1,v,'upwind')
    dcdx2 = dcdx(dx,2,v,'upwind')
    if dxmethod == 'bdf' or dxmethod == 'quick':
       templateb = - v * dcdx1 + K * dcdx2
    else:
       templateb = template

    #------------------------------------------------------------------------
    # Generate the matrix of the semi-discretized system:
    #
    #   dc
    #   -- = Ain*c
    #   dt
    #
    #------------------------------------------------------------------------
    
    # discretization matrix
    Ain = np.zeros((points,points))

    # l is the extent of the template on each side of the diagonal (for example if template= [0, a, b ,c, 0] then l=1)
    l = int((len(template)-1)/2)

    # interior points
    for i in range(l,points-l):
        Ain[i,i-l:i+l+1] = template
   
    # left boundary (note that we leave the first line blank since c0[0] is constant)   
    for i in range(1,l-1):
        # boundary template entries in the matrix
        Ain[i,0:i+l] = templateb[l-i+1:2*l+1]
        # template entries over the edge of the matrix
        for j in range(0,l-i+1):
            Ain[i,0] = Ain[i,0] + templateb[j]
    
    # right boundary    
    for i in range(points-l+1,points):
        # boundary template entries in the matrix
        Ain[i,i-2:points] = templateb[0:2+points-i]
        # template entries over the edge of the matrix
        for j in range(2+points-i,3):
            Ain[i,points-1] = Ain[i,points-1:] + templateb[j]

    #------------------------------------------------------------------------
    # Run the simulation
    #------------------------------------------------------------------------
    
    # the loop starts after the initial conditions
    if dtmethod == 'leapfrog':
        for i in range(1,timesteps-1):
            # set up the stiffness matrix and load vectors for the ith and (i-1)th timesteps
            A,b = dcdt(dt,Ain,c[i,:],c[i-1,:],dtmethod,theta)
            # solve A c(i+1) = b for c(i+1)
            c[i+1,:] = np.linalg.solve(A,b)
    else:
        for i in range(0,timesteps-1):
            # set up the stiffness matrix and load vector for the ith timestep
            A,b = dcdt(dt,Ain,c[i,:],c[i,:],dtmethod,theta)
            # solve A c(i+1) = b for c(i+1)
            c[i+1,:] = np.linalg.solve(A,b)

    return c

#------------------------------------------------------------------------
# Time integration
#------------------------------------------------------------------------
    
def dcdt(dt = None,Ain = None,cn = None,cnn = None,method = None,theta = None): 
    # Given a spatial discretization Ain, solution vectors cn and cnn, and
    # time step dt, conduct the time integration according to method
    
    n = len(Ain)

    if method == 'theta':
       # theta method: c(n+1)/dt-c(n)/dt=theta*Ain*c(n+1) + (1-theta)*Ain*c(n)
       A = np.eye(n) - dt * theta * Ain
       b = (np.eye(n) + dt * (1. - theta) * Ain).dot(cn)
    else:
       if method == 'leapfrog':
          # Leapfrog is a two-step scheme
          A = np.eye(n)
          b = cnn + 2 * dt * Ain.dot(cn)
       else:
          if method == 'Lax-Wendroff':
             # Lax-Wendroff scheme is an explicit scheme
             A = np.eye(n)
             b = (np.eye(n) + dt * Ain).dot(cn)
          else:
             if method == 'Beam-Warming':
                # Beam-Warming scheme is an explicit scheme
                A = np.eye(n)
                b = (np.eye(n) + dt * Ain).dot(cn)
             else:
                if method == 'box':
                   # theta-box method: 0.5*(c(n+1)_j+c(n+1)_(j+1)-c(n)_j-c(n)_(j+1))/dt = theta*Ain*c(n+1) + (1-theta)*Ain*c(n)
                   v = np.ones(n-1)
                   A =  0.5 * (np.eye(n) + np.diag(v,-1)) - dt * theta * Ain
                   b = (0.5 * (np.eye(n) + np.diag(v,-1)) + dt * (1. - theta) * Ain).dot(cn)
                else:
                   raise Exception(np.array(['Unknown time integration method: ',method]))

    return A,b

 #------------------------------------------------------------------------
 # Spatial discretization
 #------------------------------------------------------------------------
    
def dcdx(dx = None,n = None,v = None,method = None): 
    # Return a vector containing the interior stencil for the nth spatial derivative according to method
    # dx is the mesh width
    
    # Notes:
    #   the stencils for a method must be of odd length and centred on the main diagonal
    #   the stencils for the first and second derivative must be the same size
    
    if method == 'central':
        # second order central differences
        if n == 1:
           dc = np.array([-1./(2.*dx), 0., 1./(2.*dx)])
        else:
           dc = np.array([1./(dx**2), -2./(dx**2), 1./(dx**2)])
    else:
        if method == 'box':
           # box scheme (note that only the first derivative is supported)
           if n == 1:
              dc = np.array([-1./dx, 1./dx, 0.])
           else:
              dc = np.array([0.,0.,0.])
        else:
           if method == 'upwind':
              # first order upwind scheme
              if n == 1:
                 if v > 0:
                    dc = np.array([-1./dx, 1./dx, 0.])
                 else:
                    dc = np.array([0., -1./dx, 1./dx])
              else:
                 dc = np.array([1./(dx**2), - 2./(dx**2), 1./(dx**2)])
           else:
              if method == 'bdf':
                 # second order BDF scheme
                 if n == 1:
                    if v > 0:
                       dc = np.array([1./(2.*dx), - 4./(2.*dx), 3./(2.*dx), 0., 0.])
                    else:
                       dc = np.array([0., 0., -3./(2,*dx), 4./(2,*dx), -1./(2,*dx)])
                 else:
                    if v > 0:
                       dc = np.array([1./(dx**2), -2./(dx**2), 1./(dx**2), 0., 0.])
                    else:
                       dc = np.array([0., 0., 1./(dx**2), -2./(dx**2), 1./(dx**2)])
              else:
                 if method == 'quick':
                    # second order QUICK scheme (note that only the first derivative is supported)
                    if n == 1:
                       if v > 0:
                          dc = np.array([1./(8.*dx), -7./(8.*dx), 3./(8.*dx), 3./(8.*dx), 0.])
                       else:
                          dc = np.array([0., -3./(8.*dx), -3./(8.*dx), 7./(8.*dx), -1./(8.*dx)])
                    else:
                       dc = np.array([0.,0.,0.,0.,0.])
                 else:
                    raise Exception(np.array(['Unknown spatial discretization scheme: ',method]))

    return dc