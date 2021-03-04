import argparse
import os
import numpy as np
from models import EdwardsAnderson
from python_interface import QuantumPIAnneal

if __name__ == "__main__":

    if not os.path.exists('./results/'):
        os.mkdir('./results')
    if not os.path.exists('./results/EA'):
        os.mkdir('./results/EA')
    if not os.path.exists('./results/EA/PIQMC'):
        os.mkdir('./results/EA/PIQMC')

    # CHANGE THE DEFAULT ARGUMENTS HERE

    parser = argparse.ArgumentParser()

    parser.add_argument('--tau_schedule', default=[2**i for i in range(1,13+1)])
    parser.add_argument('--seed', default=1)
    parser.add_argument('--mcsteps', default=5) #Number of sweeps
    parser.add_argument('--P', default=20, type = int) #Number of sweeps
    parser.add_argument('--numruns', default=25, type=int)

    args = parser.parse_args()
    realization = args.seed #instance identifier
    P = args.P
    numruns = args.numruns

    nrows = 40
    ncols = 40
    gs_fname = './data/EA_'+str(nrows)+'x'+str(ncols)+'/gs_seed'+str(realization)+'.txt'
    interactions_fname = './data/'+str(nrows)+'x'+str(ncols)+'/'+str(nrows)+'x'+str(ncols)+'_uniform_seed'+str(realization)+'.txt' #load interaction instance

    model = EdwardsAnderson(nrows=nrows, ncols=ncols, gs_fname=gs_fname, interactions_fname=interactions_fname)

    Energies = np.zeros((numruns, len(args.tau_schedule),int(P)), np.float64)
    checkpointfile = './results/EA/PIQMC/EA_'+str(nrows)+'x'+str(ncols)+'_P'+str(P)+'_PIQMC_realization'+str(realization)+'_Energies.npy'
    try:
        print("Loading checkpoint!")
        Loaded = np.load(checkpointfile)
        Energies[:Loaded.shape[0]] = Loaded
    except:
        print("Failed! Running from scratch")
        Loaded = []

    for annealingrun in range(len(Loaded)+1,numruns+1):
        print("Annealing run number ", annealingrun)
        Q = QuantumPIAnneal(model, latticetype = "2D", annealingrunseed = annealingrun, **vars(args))
        Energies[annealingrun-1] = Q.perform_tau_schedule()
        np.save(checkpointfile, Energies[:annealingrun])
