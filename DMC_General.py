import numpy as np
import os,sys

class Constants:
    atomic_units = {
        "wavenumbers" : 4.55634e-6,
        "angstroms" : 1/0.529177,
        "amu" : 1.000000000000000000/6.02213670000e23/9.10938970000e-28   #1822.88839  g/mol -> a.u.
    }

    masses = {
        "H" : ( 1.00782503223, "amu"),
        "O" : (15.99491561957, "amu"),
        "D" : (2.0141017778,"amu")
    }
    @classmethod
    def convert(cls, val, unit, to_AU = True):
        vv = cls.atomic_units[unit]
        return (val * vv) if to_AU else (val / vv)

    @classmethod
    def mass(cls, atom, to_AU = True):
        m = cls.masses[atom]
        if to_AU:
            m = cls.convert(*m)
        return m


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
        disps = np.random.normal(0.0, self.sigmas, size=np.shape(walkerC)).transpose(0,2,1)
        return walkerC + disps

    def getVref(self):  # Use potential of all walkers to calculate vref
        Vbar = np.average(self.walkerV)
        if self.weighting == 'discrete':
            correction = (len(self.walkerV) - self.initialWalkers) / self.initialWalkers
        else:
            correction = (np.sum(self.contWts - np.ones(self.initialWalkers))) / self.initialWalkers
        vref = Vbar - (self.alpha * correction)
        return vref

    def propagate(self):
        DW=False
        for prop in range(self.nTimeSteps):
            if prop % 100 == 0:
                print(prop)
                print(len(self.walkerC))
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
                         atms = self.atoms
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
        np.save(self.outputFolder+"/"+self.simName+"_energies"+".npy",Constants.convert(self.vrefAr,"wavenumbers",to_AU=False))
        if self.weighting == 'discrete':
            np.save(self.outputFolder + "/" + self.simName + "_population" + ".npy",self.popAr)


if __name__ == "__main__":
    def PatrickShingle(cds):
        import subprocess as sub
        np.savetxt("PES/PES0/hoh_coord.dat", cds.reshape(cds.shape[0]*cds.shape[1],cds.shape[1]), header=str(len(cds)), comments="")
        sub.run('./calc_h2o_pot', cwd='PES/PES0')
        return np.loadtxt('PES/PES0/hoh_pot.dat')


    dmcWater = DMC(simName = "DMC_con_test",
                   outputFolder="DMCResults/",
                   weighting='continuous',
                   initialWalkers=1000,
                   nTimeSteps=1000+1,
                   equilTime=500,
                   wfnSpacing=100,
                   DwSteps=50,
                   atoms=["H", "H", "O"],
                   dimensions=3,
                   deltaT=5,
                   D=0.5,
                   potential=PatrickShingle,
                   masses=None,
                   startStructure = Constants.convert(np.array(
                    [[0.9578400,0.0000000,0.0000000],
                     [-0.2399535,0.9272970,0.0000000],
                     [0.0000000,0.0000000,0.0000000]]) * 1.01,"angstroms",to_AU=True)
                 )
    dmcWater.run()
