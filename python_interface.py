import numpy as np
import matplotlib.pyplot as plt
import piqmc.sa as sa
import piqmc.qmc as qmc
import copy

########## Simulated Quantum Annealign Class ###########

class QuantumPIAnneal():

    def __init__(self, model, latticetype = "2D", annealingrunseed = 1 , **kwargs):
        """
        Args:
            model:
            **kwargs:
        """

        #############
        # SET MODEL #
        #############

        self.model = model
        self.latticetype = latticetype

        #####################
        # QUANTUM ANNEALING #
        #####################

        self.tau_schedule = kwargs.pop('tau_schedule', [1, int((10**0.5)*1), 10, int((10**0.5)*10), 100, int((10**0.5)*100), 1000, int((10**0.5)*1000), 10000, int((10**0.5)*10000), 100000])
        print("Annealing times to be run =", self.tau_schedule)
        self.mcsteps = kwargs.pop('mcsteps', 1)
        print("num_sweeps =", self.mcsteps)
        self.gamma_0 = kwargs.pop('gamma_0', 1.0)
        print("gamma_0 = ", self.gamma_0)
        self.gamma_T = kwargs.pop('gamma_T', 1e-8)
        self.P = kwargs.pop('P', 20)
        print("P = ", self.P)
        self.PT = kwargs.pop('PT', 1.0)
        self.q_temperature = self.PT / self.P
        self.q_scheds = kwargs.pop('q_scheds',[np.linspace(self.gamma_0, self.gamma_T, t) for t in self.tau_schedule])
        print("Temperature so that P * T = 1.0:", self.q_temperature)

        ###########################
        # CLASSICAL PRE-ANNEALING #
        ###########################

        self.preannealing_temperature = kwargs.pop('preannealing_temperature', 3.0)
        self.preannealing_schedule_steps = kwargs.pop('preannealing_schedule_steps', 60)
        self.preannealing_mcsteps = kwargs.pop('preannealing_mcsteps', 100)
        self.preannealing_sched = kwargs.pop('preannealing_sched', np.linspace(self.preannealing_temperature,
                                              self.q_temperature,
                                              self.preannealing_schedule_steps))

        ##################
        # RANDOM NUMBERS #
        ##################

        self.annealingrunseed = annealingrunseed
        self.rng = np.random.RandomState(self.annealingrunseed)

        ####################
        # INITIALIZE MODEL #
        ####################

        self.spinVector = 2.0 * self.rng.randint(2, size=self.model.nspins).astype(np.float) - 1.0
        self.confs = None

    def pre_anneal(self):
        # START PRE-ANNEALING
        self.energy = []
        print("\nEnergy per spin before pre-annealing is: {}".format(
            self.model.energy(self.spinVector)/self.model.nspins))
        if self.latticetype == "2D":
            sa.Anneal(self.preannealing_sched,
                      self.preannealing_mcsteps,
                      self.spinVector,
                      self.model.nbs,
                      self.rng )
        elif self.latticetype == "FullyConnected":
            sa.AnnealFullyConnected(self.preannealing_sched,
                      self.preannealing_mcsteps,
                      self.spinVector,
                      self.model.J,
                      self.rng )
        else:
            raise Exception("The supported lattice types are either 2D or FullyConnected")

        print("Final energy per spin after pre-annealing is: {}".format(
            self.model.energy(self.spinVector)/self.model.nspins), "\n")


    def quantum_anneal(self, confs, sched):

        if self.latticetype == "2D":
            qmc.QuantumAnneal(sched,
                              self.mcsteps,
                              self.P,
                              self.q_temperature,
                              self.model.nspins,
                              confs,
                              self.model.nbs,
                              self.rng )
        elif self.latticetype == "FullyConnected":
            qmc.QuantumAnnealFullyConnected(sched,
                              self.mcsteps,
                              self.P,
                              self.q_temperature,
                              self.model.nspins,
                              confs,
                              self.model.J,
                              self.rng )
        else:
            raise Exception("The supported lattice types are either 2D or FullyConnected")

        # Get the lowest energy from all the slices
        self.minEnergy = np.inf
        Energies = []

        for col in confs:
            candidateEnergy = self.model.energy(col)
            if candidateEnergy < self.minEnergy:
                self.minEnergy = candidateEnergy
            Energies.append(candidateEnergy)

        self.Energy = Energies # 1D np array size (numtrotterslices,)

        print("Final minimal energy per spin after quantum annealing is: {}".format(self.minEnergy/self.model.nspins))
        print("Final average energy per spin after quantum annealing is: {}".format(np.mean(self.Energy)/self.model.nspins),"\n")

    def perform_tau_schedule(self):
        self.Energies = []
        self.pre_anneal()
        confs = np.tile(self.spinVector, (self.P, 1))
        for sch in self.q_scheds:
            sch_confs = copy.deepcopy(confs)
            self.quantum_anneal(sch_confs, sch)
            self.Energies.append(self.Energy)

        return np.array(self.Energies) #2D np.array with size (len(self.q_scheds), numtrotterslices)

########## Simulated Annealing Class ###########

class ClassicalAnneal():

    def __init__(self, model, latticetype = "2D", annealingrunseed = 1 , **kwargs):
        """
        Args:
            model:
            **kwargs:
        """

        #############
        # SET MODEL #
        #############

        self.model = model
        self.latticetype = latticetype

        ###########################
        # CLASSICAL ANNEALING #
        ###########################


        self.tau_schedule = kwargs.pop('tau_schedule', [1, int((10**0.5)*1), 10, int((10**0.5)*10), 100, int((10**0.5)*100), 1000, int((10**0.5)*1000), 10000, int((10**0.5)*10000), 100000])
        print("Annealing times to be run =", self.tau_schedule)
        self.mcsteps = kwargs.pop('mcsteps', 1)
        print("num_sweeps =", self.mcsteps)

        self.T0 = kwargs.pop('T_0', 1.0)
        print("T0 =", self.T0)
        self.Tf = kwargs.pop('T_f', 1e-8)
        self.T_scheds = kwargs.pop('q_scheds',[np.linspace(self.T0, self.Tf, t) for t in self.tau_schedule]) #For SA (without QA)
        self.num_warmup = kwargs.pop('num_warmup', 1000)
        print("num warmup steps =", self.num_warmup)
        ##################
        # RANDOM NUMBERS #
        ##################

        self.annealingrunseed = annealingrunseed
        self.rng = np.random.RandomState(self.annealingrunseed)

        ####################
        # INITIALIZE MODEL #
        ####################

        self.spinVector = 2.0 * self.rng.randint(2, size=self.model.nspins).astype(np.float) - 1.0
        self.confs = None

    def Anneal(self, sched):
        print("Energy per spin before warmup is: {}".format(
            self.model.energy(self.spinVector)/self.model.nspins))
        #Perform Warmup step:
        if self.latticetype == "2D":
            sa.Anneal(np.array([float(self.T0)]),
                      self.num_warmup,
                      self.spinVector,
                      self.model.nbs,
                      self.rng )
        elif self.latticetype == "FullyConnected":
            sa.AnnealFullyConnected(np.array([float(self.T0)]),
                      self.num_warmup,
                      self.spinVector,
                      self.model.J,
                      self.rng )
        else:
            raise Exception("The supported lattice types are either 2D or FullyConnected")

        print("Energy per spin after warmup is: {}".format(
            self.model.energy(self.spinVector)/self.model.nspins))
        #Perform Annealing
        if self.latticetype == "2D":
            sa.Anneal(sched,
                      self.mcsteps,
                      self.spinVector,
                      self.model.nbs,
                      self.rng )
        elif self.latticetype == "FullyConnected":
            sa.AnnealFullyConnected(sched,
                      self.mcsteps,
                      self.spinVector,
                      self.model.J,
                      self.rng )
        else:
            raise Exception("The supported lattice types are either 2D or FullyConnected")

        print("Final energy per spin after annealing is: {}".format(
            self.model.energy(self.spinVector)/self.model.nspins), "\n")
        self.Energy = self.model.energy(self.spinVector)

    def perform_tau_schedule(self):
        self.Energies = []
        for sch in self.T_scheds:
            self.spinVector = 2.0 * self.rng.randint(2, size=self.model.nspins).astype(np.float) - 1.0
            self.Anneal(sch)
            self.Energies.append(self.Energy)
        return np.array(self.Energies) #2D np.array with size (len(self.T_scheds))
