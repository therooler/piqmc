import numpy as np
import matplotlib.pyplot as plt
import piqmc.sa as sa
import piqmc.qmc as qmc
import copy


class QuantumPIAnneal():

    def __init__(self, model, experiment_name, **kwargs):
        """

        Args:
            model:
            experiment_name:
            **kwargs:
        """


        #############
        # SET MODEL #
        #############

        self.model = model
        self.experiment_name = experiment_name

        #####################
        # QUANTUM ANNEALING #
        #####################

        self.tau_schedule = kwargs.pop('tau_schedule', [10, 100, 1000, 10000])
        self.mcsteps = kwargs.pop('mcsteps', 1)
        # tau = mcsteps * q_annealing_steps
        self.gamma_0 = kwargs.pop('gamma_0', 3.0)
        self.gamma_T = kwargs.pop('gamma_T', 1e-8)
        self.P = kwargs.pop('P', 40)
        self.PT = kwargs.pop('PT', 1.0)
        self.q_temperature = self.PT / self.P
        self.q_scheds = kwargs.pop('q_scheds',[np.linspace(self.gamma_0, self.gamma_T, self.mcsteps * t) for t in self.tau_schedule])
        self.TAU_SCHED = False
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

        seed = kwargs.pop('seed', None)
        self.rng = np.random.RandomState(seed)

        ####################
        # INITIALIZE MODEL #
        ####################

        self.spinVector = 2.0 * self.rng.randint(2, size=self.model.nspins).astype(np.float) - 1.0
        self.confs = None

    def pre_anneal(self):
        # START PRE-ANNEALING
        self.energy = []
        print("\nEnergy per site before pre-annealing is: {}".format(
            self.model.energy(self.spinVector) / self.model.nspins))
        sa.Anneal(self.preannealing_sched,
                  self.preannealing_mcsteps,
                  self.spinVector,
                  self.model.nbs,
                  self.rng )
        print("\nFinal energy per site after pre-annealing is: {}".format(
            self.model.energy(self.spinVector) / self.model.nspins))

    def quantum_anneal(self, confs, sched):

        qmc.QuantumAnneal(sched,
                          self.mcsteps,
                          self.P,
                          self.q_temperature,
                          self.model.nspins,
                          confs,
                          self.model.nbs,
                          self.rng )
        # Get the lowest energy from all the slices
        self.minEnergy = np.inf
        for col in confs:
            candidateEnergy = self.model.energy(col) / self.model.nspins
            if candidateEnergy < self.minEnergy:
                self.minEnergy = candidateEnergy
        print("\nFinal energy per site after quantum annealing is: {}".format(self.minEnergy))

    def perform_tau_schedule(self):
        self.residual_energy = []
        self.pre_anneal()
        confs = np.tile(self.spinVector, (self.P, 1))
        for sch in self.q_scheds:
            sch_confs = copy.deepcopy(confs)
            self.quantum_anneal(sch_confs, sch)
            self.residual_energy.append(abs(self.minEnergy - self.model.gsenergy))
        self.TAU_SCHED = True

    def save_results(self):
        assert self.TAU_SCHED, 'Run "perform_tau_schedule" first.'
        params = {'tau_schedule': self.tau_schedule, 'mcsteps': self.mcsteps, 'gamma_0': self.gamma_0,
                  'gamma_T': self.gamma_T, 'P': self.P, 'PT': self.PT, 'q_temperature': self.q_temperature,
                  'q_scheds': self.q_scheds, 'preannealing_temperature': self.preannealing_temperature,
                  'preannealing_schedule_steps': self.preannealing_schedule_steps,
                  'preannealing_mcsteps': self.preannealing_mcsteps,
                  'preannealing_schedchedule': self.preannealing_sched}

        np.save('./results/' + self.experiment_name + '_res_e', self.residual_energy)
        np.save('./results/' + self.experiment_name + '_tau_sched', self.tau_schedule)
        with open('./results/' + self.experiment_name + '_tau_sched.txt', 'w') as file:
            for k, v in params.items():
                file.write(str(k) + '=' + str(v) + '\n')

    def plot_results_tau_schedule(self):
        fig1 = plt.figure(1)
        ax1 = fig1.gca()
        ax1.set_title("")
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_ylim([0.1*np.min(self.residual_energy), 1])
        ax1.set_ylabel("Residual energy per site")
        ax1.set_xlabel(r"$\tau$")
        plt.plot(self.tau_schedule, self.residual_energy)
        plt.savefig('./results/figures/' + self.experiment_name + '_res_e_tau_sched' + '.pdf')

        plt.show()
