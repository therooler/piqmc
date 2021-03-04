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

        self.nrows = nrows
        self.ncols = ncols
        self.nspins = nrows * ncols

        loaded = np.loadtxt(interactions_fname)
        self.J = sps.dok_matrix((self.nspins, self.nspins))
        for i, j, val in loaded:
            self.J[i - 1, j - 1] = val

        ###############
        # SPIN SYSTEM #
        ###############

        self.nbs = generate_neighbors(self.nspins, self.J, 4)
        self.J = self.J.toarray()

    def energy(self, spins):
        return np.dot(spins, -self.J.dot(spins))

    def energy_parallel(self, samples):
        samples_zero_or_one = 0.5*(samples+1)
        samples_reshaped = np.transpose(samples_zero_or_one.reshape([-1, self.nrows, self.ncols]), axes = (0,2,1))
        return Ising2D_diagonal_matrixelements(self.Jz, self.nrows, self.ncols, samples_reshaped)


class Wishart():
    def __init__(self, nspins, interactions):
        #####################
        # SPIN GLASS SERVER #
        #####################

        self.nspins = nspins

        self.J = interactions

        ###############
        # SPIN SYSTEM #
        ###############

        self.gsenergy = self.energy(np.ones(self.nspins)) #planted solution is all spins up or all spins down
        print("True groundstate energy per spin: ", self.gsenergy/self.nspins)

    def energy(self, spins):
        return np.dot(spins, -self.J.dot(spins))/2

class SK():
    def __init__(self, nspins, interactions_fname):
        #####################
        # SPIN GLASS SERVER #
        #####################

        self.nspins = nspins

        loaded = np.loadtxt(interactions_fname)
        self.J = np.zeros((self.nspins, self.nspins))
        for i, j, val in loaded:
            self.J[int(i) - 1, int(j) - 1] = val

        # Symmetrization of the adjacency matrix
        for i in range(self.nspins):
            for j in range(i):
                self.J[i,j] = self.J[j,i]

        print("Couplings:", self.J)

    def energy(self, spins):
        return np.dot(spins, -self.J.dot(spins))/2
