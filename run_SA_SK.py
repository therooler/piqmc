import argparse
import os
import numpy as np
from models import SK
from python_interface import ClassicalAnneal

if __name__ == "__main__":

    if not os.path.exists('./results/'):
        os.mkdir('./results')
    if not os.path.exists('./results/SK'):
        os.mkdir('./results/SK')
    if not os.path.exists('./results/SK/SA'):
        os.mkdir('./results/SK/SA')

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=1)
    parser.add_argument('--tau_schedule', default=[2**i for i in range(1,14+1)])
    parser.add_argument('--T_0', default=2.0, type=float)
    parser.add_argument('--mcsteps', default=5) #Number of sweeps
    parser.add_argument('--num_warmup', type = int, default=2000) #Number of warmup steps
    parser.add_argument('--numruns', default=50, type=int)

    args = parser.parse_args()
    realization = args.seed
    num_warmup = args.num_warmup
    numruns = args.numruns

    N = 100

    interactions_fname = './data/SK_N'+str(N)+'/'+str(N)+'_SK_seed'+str(realization)+'.txt'

    model = SK(nspins=N, interactions_fname=interactions_fname)

    Energies = np.zeros((numruns, len(args.tau_schedule)), np.float64)
    checkpointfile = './results/SK/SA/SK_N'+str(N)+'_SA_realization'+str(realization)+'_numwarmup'+str(num_warmup)+'_Energies.npy'
    try:
        print("Loading checkpoint!")
        Loaded = np.load(checkpointfile)
        Energies[:Loaded.shape[0]] = Loaded
    except:
        print("Failed! Running from scratch")
        Loaded = []

    for annealingrun in range(len(Loaded)+1,numruns+1):
        print("Annealing run number ", annealingrun)
        SA = ClassicalAnneal(model, latticetype = "FullyConnected", annealingrunseed = annealingrun, **vars(args))
        Energies[annealingrun-1] = SA.perform_tau_schedule()
        np.save(checkpointfile, Energies[:annealingrun])
