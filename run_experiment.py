import argparse
from models import EdwardsAnderson
from python_interface import QuantumPIAnneal

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ###################
    # ADD KWARGS HERE #
    ###################

    parser.add_argument('--P', default=100)

    args = parser.parse_args()

    nrows = 20
    ncols = 20
    gs_fname = './data/20x20/gs_seed1.txt'
    interactions_fname = './data/20x20/20x20_uniform_seed1.txt'

    model = EdwardsAnderson(nrows=nrows, ncols=ncols, gs_fname=gs_fname, interactions_fname=interactions_fname)

    Q = QuantumPIAnneal(model, experiment_name='test_1', **vars(args))
    Q.perform_tau_schedule()
    Q.save_results()
    Q.plot_results_tau_schedule()
