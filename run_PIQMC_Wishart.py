import argparse
import os
import numpy as np
from models import Wishart
from python_interface import QuantumPIAnneal

if __name__ == "__main__":

    if not os.path.exists('./results/'):
        os.mkdir('./results')
    if not os.path.exists('./results/Wishart'):
        os.mkdir('./results/Wishart')
    if not os.path.exists('./results/Wishart/PIQMC'):
        os.mkdir('./results/Wishart/PIQMC')

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=1)
    parser.add_argument('--P', default=100)
    parser.add_argument('--tau_schedule', default=[2**i for i in range(1,14+1)])
    parser.add_argument('--alpha', default=0.5)
    parser.add_argument('--mcsteps', default=5) #Number of sweeps
    parser.add_argument('--numruns', default=50, type=int)

    args = parser.parse_args()
    realization = args.seed
    P = args.P
    numruns = args.numruns


    N = 32
    alpha = args.alpha


    interactions_fname = './data/wishart_N'+str(N)+'/wpe_size'+str(N)+'_alpha'+str(alpha)+'_realization'+str(realization)+'.txt'

    Jz = np.loadtxt(interactions_fname)
    model = Wishart(nspins=N, interactions=Jz)

    Energies = np.zeros((numruns, len(args.tau_schedule),int(P)), np.float64)
    checkpointfile = './results/Wishart/PIQMC/Wishart_N'+str(N)+'_alpha'+str(alpha)+'_PIQMC_realization'+str(realization)+'_Energies.npy'
    try:
        print("Loading checkpoint!")
        Loaded = np.load(checkpointfile)
        Energies[:Loaded.shape[0]] = Loaded
    except:
        print("Failed! Running from scratch")
        Loaded = []

    for annealingrun in range(len(Loaded)+1, numruns+1):
        print("annealing run = ", annealingrun)
        Q = QuantumPIAnneal(model, latticetype = "FullyConnected", annealingrunseed = annealingrun,  **vars(args))
        Energies[annealingrun-1] = Q.perform_tau_schedule()
        np.save(checkpointfile, Energies[:annealingrun])
