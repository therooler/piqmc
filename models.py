import numpy as np
import scipy.sparse as sps


def generate_neighbors(nspins, J, maxnb):
    # the neighbors data structure
    nbs = np.zeros((nspins, maxnb, 2))
    # Iterate over all spins
    for ispin in range(nspins):
        ipair = 0
        # Find the pairs including this spin
        for pair in list(J.keys()):
            if pair[0] == ispin:
                nbs[ispin, ipair, 0] = pair[1]
                nbs[ispin, ipair, 1] = J[pair]
                ipair += 1
            elif pair[1] == ispin:
                nbs[ispin, ipair, 0] = pair[0]
                nbs[ispin, ipair, 1] = J[pair]
                ipair += 1
    return nbs


class EdwardsAnderson():
    def __init__(self, nrows, ncols, gs_fname, interactions_fname):
        #####################
        # SPIN GLASS SERVER #
        #####################

        self.nspins = nrows * ncols
        self.groundstate = -np.ones(self.nspins)
        self.gsspinups = np.loadtxt(gs_fname).astype(int) - 1

        self.groundstate[self.gsspinups] = 1
        loaded = np.loadtxt(interactions_fname)
        self.J = sps.dok_matrix((self.nspins, self.nspins))
        for i, j, val in loaded:
            self.J[i - 1, j - 1] = val

        self.gsenergy = self.energy(self.groundstate) / self.nspins
        print("True groundstate energy per site: ", self.gsenergy)

        ###############
        # SPIN SYSTEM #
        ###############

        self.nbs = generate_neighbors(self.nspins, self.J, 4)
        self.J = self.J.tocsc()

    def energy(self, spins):
        return np.dot(spins, -self.J.dot(spins))
