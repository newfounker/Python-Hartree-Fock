
import numpy as np

# This class has all the routines to perform UHF calculations on a Hubbard Hamiltonian

class Hubb:
    
    # Values on call
    # Nx    : Length of the Lattice
    # Ny    : Width of the lattice
    # NOccA : Number of spin up electrons
    # NoccB : Number of spin down electrons
    # U     : On-site repulsion
    # tol   : Maximum norm of the residual allowed at convergence
    
    def __init__(self,Nx,Ny,NOccA,NOccB,U,tol):

        self.Nx       = Nx
        self.Ny       = Ny
        self.NAO      = Nx*Ny
        self.NOccA    = NOccA
        self.NOccB    = NOccB
        self.NOcc     = NOccA+NOccB
        self.U        = U
        self.tol      = tol
        self.err      = 0.0
        self.DenA     = np.zeros((Nx*Ny,Nx*Ny))
        self.DenB     = np.zeros((Nx*Ny,Nx*Ny))
        self.FockA    = np.zeros((Nx*Ny,Nx*Ny))
        self.FockB    = np.zeros((Nx*Ny,Nx*Ny))
        self.T        = np.zeros((Nx*Ny,Nx*Ny))
        self.EvecA    = np.zeros((Nx*Ny,Nx*Ny))
        self.EvecB    = np.zeros((Nx*Ny,Nx*Ny))
        self.Energy   = 0.0
        self.errVec   = np.zeros((2*self.NAO*self.NAO))
        self.DIISset  = False
        
# Define the hopping matrix

    # doPBCx : True => periodic boundary conditions in x direction
    # doPBCy : True => periodic boundary conditions in y direction
        
    def mkHopping(self,doPBCx,doPBCy):
        
        if self.Nx == 1 or self.Ny == 1:
            
            # This will construct hopping for a 1D system.
            
            self.T[:-1,1:] = np.diag(-np.ones(self.NAO-1))
            
            #Check the periodicity and include.
            
            if self.Nx > self.Ny:
                tmp = doPBCx
            else:
                tmp = doPBCy
            
            if tmp:
                self.T[0,-1] = -1.0
            
            self.T = self.T + self.T.T
            
        else:
        
            # This will construct hopping for a 2D system.
        
            for i in range(self.Nx-1):
                ih = i + 1
                for j in range(self.Ny):
                    self.T[i+j*self.Nx,ih+j*self.Nx] += -1.0
                
            for i in range(1,self.Nx):
                ih = i - 1
                for j in range(self.Ny):
                    self.T[i+j*self.Nx,ih+j*self.Nx] += -1.0            
                            
            for j in range(self.Ny-1):
                jh = j + 1
                for i in range(self.Nx):
                    self.T[i+j*self.Nx,i+jh*self.Nx] += -1.0
                            
            for j in range(1,self.Ny):
                jh = j - 1
                for i in range(self.Nx):
                    self.T[i+j*self.Nx,i+jh*self.Nx] += -1.0
            
            # Add in the periodicity
            
            if doPBCx:
                i  = self.Nx-1
                ih = 0
                for j in range(self.Ny):
                    self.T[i+j*self.Nx,ih+j*self.Nx] += -1.0
                i  = 0
                ih = self.Nx-1
                for j in range(self.Ny):
                    self.T[i+j*self.Nx,ih+j*self.Nx] += -1.0
                            
            if doPBCy:
                j  = self.Ny-1
                jh = 0
                for i in range(self.Nx):
                    self.T[i+j*self.Nx,i+jh*self.Nx] += -1.0
                j  = 0
                jh = self.Ny-1
                for i in range(self.Nx):
                    self.T[i+j*self.Nx,i+jh*self.Nx] += -1.0
        
        
        
    # Construct a guess state
    # addCor  : portion of guess that is U=0   solution
    # addNeel : portion of guess that is U=Inf solution
        
    def mkGuess(self,addCore=0.0,addNeel=1.0):
        
        self.EvecA.fill(0.)
        self.EvecB.fill(0.)
        
        # These will keep track of the position in each lattice.
        Ko = 0
        Kv = 0
        
        # Go throught the lattice alternating adding to either the 
        # up or down electrons in a checkerboard pattern.
        # Note to self:
        #    Maybe come back and redo this with list comprehension.
        for i in range(self.Nx):
            for j in range(self.Ny):
                if ((i+j) % 2) == 1:
                    self.EvecA[i+self.Nx*j,Ko]              = 1.0
                    self.EvecB[i+self.Nx*j,Ko+self.NAO//2]  = 1.0
                    Ko += 1
                else:
                    self.EvecA[i+self.Nx*j,Kv+self.NAO//2]  = 1.0
                    self.EvecB[i+self.Nx*j,Kv]              = 1.0
                    Kv += 1
        
        # Get the plane wave vectors from the core guess.
        e,v = np.linalg.eig(self.T)
        e = np.argsort(e.real)
        v = v[:,e]
        
        # Add the results together. We do not need to orthogonalize 
        # here as that will be taken care of by the SCF cycles later 
        # anyway.
        self.EvecA = addNeel*self.EvecA + addCore*v
        self.EvecB = addNeel*self.EvecB + addCore*v
                     
    
    # Construct the density matricies out of the orbital coefficients
    
    def newDen(self):
        self.DenA = np.matmul(self.EvecA[:,:self.NOccA],self.EvecA[:,:self.NOccA].T)
        self.DenB = np.matmul(self.EvecB[:,:self.NOccB],self.EvecB[:,:self.NOccB].T)
                     
    # Calculate the energy
    
    def getEnergy(self):
        self.Energy = np.trace(np.matmul(self.T,self.DenA+self.DenB)) \
                              +self.U*np.dot(np.diag(self.DenA),np.diag(self.DenB))


    # Construct the Fock matrix
    # damping : damping parameter to controll convergence
    # doUHF   : flag to determine weather the calculation will be RHF or UHF

    def mkFock(self,damping,doUHF):
        
        # Add the Hopping component
        self.FockA = damping*self.FockA + self.T
        self.FockB = damping*self.FockB + self.T
        
        # Add the double occupancy component
        for i in range(self.NAO):
            self.FockA[i,i] += self.U*self.DenB[i,i]
            self.FockB[i,i] += self.U*self.DenA[i,i]
            
        # If doing RHF, average the up and down components
        if not doUHF:
            self.FockA += self.FockB/2.0
            self.FockB = 0.0+self.FockA
        
    # Diagonalize the Fock matrix to construct the orbitals 
    # for the next iteration.
        
    def updateEvecs(self):
        
        # Diagonalize the up spins
        e,self.EvecA = np.linalg.eigh(self.FockA)
        
        # Sort to select the smallest eigenvalues
        e = np.argsort(e.real)
        self.EvecA = self.EvecA[:,e]
        
        # Repeat for the down spins
        e,self.EvecB = np.linalg.eigh(self.FockB)
        e = np.argsort(e.real)
        self.EvecB = self.EvecB[:,e]
        
    # Calculate the residual: R = [Fock,Den]
        
    def geterr(self):
        
        # Evaluate the commutators.
        self.errVec[:(self.NAO*self.NAO)] = np.reshape(np.matmul(self.FockA,self.DenA )  \
                                - np.matmul(self.DenA,self.FockA),self.NAO*self.NAO)
        self.errVec[(self.NAO*self.NAO):] = np.reshape(np.matmul(self.FockB,self.DenB )  \
                                - np.matmul(self.DenB,self.FockB),self.NAO*self.NAO)
        
        # Get the norm to determine if we are converged.
        self.err = np.linalg.norm(self.errVec)
        
        
    # Initialize DIIS variables
    # NDIIS: Number of previous iterations to save for the fitting
    # Ninit: Number of iterations to wait before convergence accleration
    # Note: Must keep NDIIS <= Ninit
        
    def setupDIIS(self,NDIIS=3,Ninit=5):
        self.NDIIS = NDIIS
        self.Ninit = min(Ninit,NDIIS) 
        # In future I should allow a variable number so the number of used 
        # iterations can increase as we calculate more.
        
        # Create arrays to store NDIIS previous iterations
        self.FockADIISList = np.zeros((self.NAO,self.NAO,NDIIS))
        self.FockBDIISList = np.zeros((self.NAO,self.NAO,NDIIS))
        self.errVecs       = np.zeros((2*self.NAO*self.NAO,NDIIS))
        
        # Indicate that DIIS is set to go.
        self.DIISset = True
        
    
    # Add the current iteration to the DIIS history.
        
    def updateDIIS(self):
        
        # It would probably be more efficient to cycle through the 
        # index for the latest iteration rather than shuffling 
        # everything around. I may come back to fix this in the future.
        self.errVecs[:,:-1] = self.errVecs[:,1:]
        self.errVecs[:,-1]  = self.errVec
        self.FockADIISList[:,:,:-1] = self.FockADIISList[:,:,1:]
        self.FockBDIISList[:,:,:-1] = self.FockBDIISList[:,:,1:]
        self.FockADIISList[:,:,-1] = self.FockA
        self.FockBDIISList[:,:,-1] = self.FockB
        
        
    # Construct the New Fock matrix from the current iteration and the history
    def getFockDIIS(self):
        
        # This is the DIIS fitting.
        vec = np.zeros(self.NDIIS+1)
        B = np.ones((self.NDIIS+1,self.NDIIS+1))
        vec[-1] = 1.
        B[-1,-1] = 0.
        B[:self.NDIIS,:self.NDIIS] = \
            np.array([[np.dot(self.errVecs[:,i],self.errVecs[:,j]) \
                       for i in range(self.NDIIS)] for j in range(self.NDIIS)])
        vec = np.linalg.solve(B,vec)
        
        # Now using the DIIS results, construct the new Fock matrix.
        self.FockA = np.zeros((self.NAO,self.NAO))
        self.FockB = np.zeros((self.NAO,self.NAO))
        for i in range(self.NDIIS):
            self.FockA += self.FockADIISList[:,:,i]*vec[i]
            self.FockB += self.FockBDIISList[:,:,i]*vec[i]
        
        
    # Perform the SCF cycles untill convergence.
    # damping: add a damping paramater for the cycles (may help unstable systems)
    # doUHF  : flag to determine if the calculation is RHF or UHF
    # doDIIS : flag to determine weather or not DIIS convergence is used,
    #          setupDIIS must be called first for this to function
        
    def doSCF(self, damping=0.0, doUHF = True, doDIIS = False):
        
        self.err = 1.0
        niter = 0
        self.newDen()
        
        if doDIIS and not self.DIISset:
            # Put an error statement here.
            error('setupDIIS must be called before doSCF for DIIS convergence acceleration to work!')
        
        while self.tol < self.err:
            self.mkFock(damping,doUHF)
            self.geterr()
            if doDIIS:
                self.updateDIIS()
                if niter >= self.Ninit:
                    if niter == self.Ninit:
                        print('Starting DIIS convergence acceleration.')
                    self.getFockDIIS()
            self.updateEvecs()
            self.newDen()
            self.getEnergy()
            niter += 1
            print('Iteration:',niter,'Energy:',self.Energy,'Residual:',self.err)

