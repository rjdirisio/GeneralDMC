import numpy as np
import os
from researchUtils import Constants

class DMC:
    def __init__(self,
                 simName = "DMC_Sim",
                 outputFolder = "DMCResults/",
                 weighting = 'discrete',
                 initialWalkers = 1000,
                 nTimeSteps = 10000,
                 equilTime = 2000,
                 wfnSpacing = 1000,
                 DwSteps = 50,
                 atoms = [],
                 dimensions = 1,
                 deltaT = 5,
                 D = 0.5,
                 potential=None,
                 masses = None,
                 startStructure = None
                 ):
        """
        :param simName:Simulation name for saving wavefunctions
        :type simName:str
        :param outputFolder:The folder where the results will be stored, including wavefunctions and energies
        :type outputFolder:str
        :param weighting:Discrete or Continuous weighting DMC.  Continuous means that there are fixed number of walkers
        :type weighting:str
        :param initialWalkers:Number of walkers we will start the simulation with
        :type initialWalkers:int
        :param nTimeSteps:Total time steps we will be propagating the walkers.  nTimeSteps*deltaT = total time in A.U.
        :type nTimeSteps:int
        :param equilTime: Time before we start collecting wavefunctions
        :type equilTime:int
        :param wfnSpacing:How many time steps in between we will propagate before collecting another wavefunction
        :type wfnSpacing:int
        :param DwSteps:Number of time steps for descendant weighting.
        :type DwSteps: int
        :param atoms:List of atoms for the simulation
        :type atoms:list
        :param dimensions: 3 leads to a 3N dimensional simulation. This should always be 3 for real systems.
        :type dimensions:int
        :param deltaT: The length of the time step; how many atomic units of time are you going in one time step.
        :type deltaT: int
        :param D: Diffusion Coefficient.  Usually set at 0.5
        :type D:float
        :param potential: Takes in coordinates, gives back energies
        :type potential: function
        :param masses:For feeding in artificial masses in atomic units.  If not, then the atoms param will designate masses
        :type masses: list
        :param startStructure:An initial structure to initialize all your walkers
        :type startStructure:np.ndarray
        """
        self.atoms=atoms
        self.simName = simName
        self.outputFolder = outputFolder
        self.initialWalkers = initialWalkers
        self.nTimeSteps = nTimeSteps
        self.potential = potential
        self.weighting = weighting
        self.DwSteps=DwSteps
        self.WfnSaveStep = np.arange(equilTime,nTimeSteps,wfnSpacing)
        self.DwSaveStep = self.WfnSaveStep+self.DwSteps
        self.whoFrom = None #Not descendant weighting yet
        self.walkerV = np.zeros(self.initialWalkers)
        self.vrefAr = np.zeros(self.nTimeSteps)
        self.popAr = np.zeros(self.nTimeSteps)
        self.deltaT = deltaT
        self.alpha = 1.0 / (2.0 * deltaT)  # simulation parameter - adjustable

        if startStructure is None:
            self.walkerC = np.zeros(self.initialWalkers,len(atoms),dimensions)
        else:
            self.walkerC = np.repeat(np.expand_dims(startStructure, axis=0), self.initialWalkers, axis=0)
        if masses is None:
            masses = np.array([ Constants.mass(a) for a in self.atoms ])
        self.sigmas = np.sqrt((2 * D * deltaT) / masses)
        if not os.path.isdir(self.outputFolder):
            os.makedirs(self.outputFolder)
        if self.weighting == 'continuous':
            self.contWts = np.ones(self.initialWalkers)
        else:
            self.contWts = None

    def birthOrDeath_vec(self,vref, Desc):
        """
        Chooses whether or not the walker made a bad enough random walk to be removed from the simulation.
        For discrete weighting, this leads to removal or duplication of the walkers.  For continuous, this leads
         to an update of the weights and a potential branching of a large weight walker to the smallest one
         """
        if self.weighting == 'discrete':
            randNums = np.random.random(len(self.walkerC))
            deathMask = np.logical_or((1 - np.exp(-1. * (self.walkerV - vref) * self.deltaT)) < randNums, self.walkerV < vref)
            self.walkerC = self.walkerC[deathMask]
            self.walkerV = self.walkerV[deathMask]
            randNums = randNums[deathMask]
            if Desc:
                self.whoFrom = self.whoFrom[deathMask]

            birthMask = np.logical_and((np.exp(-1. * (self.walkerV - vref) * self.deltaT) - 1) > randNums, self.walkerV < vref)
            self.walkerC = np.concatenate((self.walkerC, self.walkerC[birthMask]))
            self.walkerV = np.concatenate((self.walkerV, self.walkerV[birthMask]))
            if Desc:
                self.whoFrom = np.concatenate((self.whoFrom, self.whoFrom[birthMask]))
            return self.whoFrom,self.walkerC,self.walkerV
        else:
            self.contWts = self.contWts*np.exp(-1.0*(self.walkerV - vref) * self.deltaT)
            thresh = 1.0/self.initialWalkers
            killMark = np.where(self.contWts < thresh)[0]
            for walker in killMark:
                maxWalker = np.argmax(self.contWts)
                self.walkerC[walker] = np.copy(self.walkerC[maxWalker])
                self.walkerV[walker] = np.copy(self.walkerV[maxWalker])
                if Desc:
                    self.whoFrom[walker]=self.whoFrom[maxWalker]
                self.contWts[maxWalker] /= 2.0
                self.contWts[walker] = np.copy(self.contWts[maxWalker])
            return self.contWts,self.whoFrom,self.walkerC,self.walkerV

    def moveRandomly(self,walkerC):
        disps = np.random.normal(0.0, self.sigmas, size=np.shape(walkerC.transpose(0,2,1))).transpose(0,2,1)
        return walkerC + disps

    def getVref(self):  # Use potential of all walkers to calculate vref
        """
             Use the energy of all walkers to calculate vref with a correction for the fluctuation in the population
             or weight.
         """
        Vbar = np.average(self.walkerV)
        if self.weighting == 'discrete':
            correction = (len(self.walkerV) - self.initialWalkers) / self.initialWalkers
        else:
            correction = (np.sum(self.contWts - np.ones(self.initialWalkers))) / self.initialWalkers
        vref = Vbar - (self.alpha * correction)
        return vref

    def propagate(self):
        """
             The main DMC loop.
             1. Move Randomly
             2. Calculate the Potential Energy
             3. Birth/Death
             4. Update Vref
             Additionally, checks when the wavefunction has hit a point where it should save / start descendent
             weighting.
         """
        DW=False
        for prop in range(self.nTimeSteps):
            if prop % 100 == 0:
                print(f'propagation step {prop}')
                if self.weighting == 'discrete':
                    print(f'num walkers : {len(self.walkerC)}')
            self.walkerC = self.moveRandomly(self.walkerC)
            self.walkerV = self.potential(self.walkerC)
            if prop == 0:
                Vref = self.getVref()
            if prop in self.WfnSaveStep:
                dwts = np.zeros(len(self.walkerC))
                parent = np.copy(self.walkerC)
                self.whoFrom = np.arange(len(self.walkerC))
                DW = True
            if prop in self.DwSaveStep:
                DW = False
                if self.weighting == 'discrete':
                    unique, counts = np.unique(self.whoFrom, return_counts=True)
                    dwts[unique]=counts
                else:
                    for q in range(len(self.contWts)):
                        dwts[q] = np.sum(self.contWts[self.whoFrom == q])
                np.savez(self.outputFolder+"/"+self.simName+"_wfn_"+str(prop-self.DwSteps)+"ts",
                         coords=parent,
                         weights=dwts,
                         nDw = self.DwSteps,
                         atms = self.atoms,
                         vref=self.vrefAr
                         )
            if self.weighting=='discrete':
                self.whoFrom, self.walkerC, self.walkerV = self.birthOrDeath_vec(Vref,DW)
            else:
                self.contWts,self.whoFrom, self.walkerC, self.walkerV = self.birthOrDeath_vec(Vref, DW)
            Vref = self.getVref()
            self.vrefAr[prop] = Vref
            self.popAr[prop] = len(self.walkerC)
    def run(self):
        self.propagate()
        np.save(self.outputFolder+"/"+self.simName+"_energies.npy",Constants.convert(self.vrefAr,"wavenumbers",to_AU=False))
        if self.weighting == 'discrete':
            np.save(self.outputFolder + "/" + self.simName + "_population" + ".npy",self.popAr)


if __name__ == "__main__":
    def PatrickShingle(cds):
        import subprocess as sub
        np.savetxt("PES/PES0/hoh_coord.dat", cds.reshape(cds.shape[0]*cds.shape[1],cds.shape[-1]), header=str(len(cds)), comments="")
        sub.run('./calc_h2o_pot', cwd='PES/PES0')
        return np.loadtxt('PES/PES0/hoh_pot.dat')

    # def protCluster(cds):
    #     atms = ['H','H','H','O','H','H','O','H','H','O']
    #     import subprocess as sub
    #     import multiprocessing as mp
    #     splt = np.array_split(cds,mp.cpu_count())
    #     for k in range(mp.cpu_count()):
    #         tmm2 = time.time()
    #         fllK = open('big'+str(k)+'coord.dat','w+')
    #         fllK.write("10\n")
    #         fllK.write("%d\n" % len(splt[k]))
    #         for walk in range(len(splt[k])):
    #             for atm in range(len(atms)):
    #                 fllK.write("%0.18f %0.18f %0.18f %s\n" % (splt[k][walk,atm,0],splt[k][walk,atm,1],splt[k][walk,atm,2],atms[atm]))
    #         fllK.close()
    #     print(f'THAT took {time.time() - tmm2} seconds.')
    #     sub.call('runPots.sh')
    #     vprime = np.loadtxt("big1/eng_dip.dat")[:,0]
    #     for k in range(2,mp.cpu_count+1):
    #         v = np.concatenate(vprime,np.loadtxt("big"+str(k)+"/eng_dip.dat")[:,0])
    #     return v
    def HODMC(cds):
        omega = Constants.convert(3000.,'wavenumbers',to_AU=True)
        mass = Constants.mass('H',to_AU=True)
        return np.squeeze(0.5*mass*omega**2*cds**2)

    dmc_HO = DMC(simName = "DMC_con_test",
                   outputFolder="~/HODMC/",
                   weighting='discrete',
                   initialWalkers=10000,
                   nTimeSteps=10000+1,
                   equilTime=1000,
                   wfnSpacing=5000,
                   DwSteps=50,
                   atoms=['H'],
                   dimensions=1,
                   deltaT=5,
                   D=0.5,
                   potential=HODMC,
                   masses=None,
                   startStructure = Constants.convert(
                       np.array([[0.00000]]),"angstroms",to_AU=True))
    dmc_HO.run()
    # dmcTrimer = DMC(simName = "DMC_con_test",
    #                outputFolder="DMCResults/",
    #                weighting='discrete',
    #                initialWalkers=1000,
    #                nTimeSteps=1000+1,
    #                equilTime=500,
    #                wfnSpacing=100,
    #                DwSteps=50,
    #                atoms=['H','H','H','O','H','H','O','H','H','O'],
    #                dimensions=3,
    #                deltaT=5,
    #                D=0.5,
    #                potential=protCluster,
    #                masses=None,
    #                startStructure = Constants.convert(
    #                    np.array([[0.00000, 0.91527, -0.05817],
    #                        [0.00000, 1.67720, 0.53729],
    #                        [-0.87302, 0.35992, 0.01707],
    #                        [2.56267, -0.75858, 0.76451],
    #                        [2.70113, -0.40578, -0.73813],
    #                        [2.07091, -0.46191, -0.00993],
    #                        [0.87302, 0.35993, 0.01707],
    #                        [-2.70115, -0.40575, -0.73811],
    #                        [-2.56265, -0.75862, 0.76451],
    #                        [-2.07092, -0.46190, -0.00993]]
    #                             )* 1.01
    #                    ,"angstroms",to_AU=True))
    # dmcTrimer.run()
