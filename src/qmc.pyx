cimport cython
cimport numpy as np

from tqdm import tqdm
from libc.math cimport exp as cexp
from libc.math cimport tanh as ctanh
from libc.math cimport log as clog
from libc.stdlib cimport rand as crand
from libc.stdlib cimport RAND_MAX as RAND_MAX


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef QuantumAnneal(np.float_t[:] sched,
                    int mcsteps,
                    int slices,
                    float temp,
                    int nspins,
                    np.float_t[:, :] confs,
                    np.float_t[:, :, :] nbs,
                    rng):
    """
    Adapted from Hadayat Seddiqi's code see: https://github.com/hadsed/pathintegral-qmc/

    Perform quantum annealing using path-integral quantum Monte Carlo.

    Args:
        @sched (np.array, float): an array of temperatures that specify
                                  the annealing schedule
        @mcsteps (int): number of sweeps to do on each annealing step
        @slices (int): number of replicas
        @temp (float): temperature after pre-annealing
        @temp (float): ambient temperature
        @confs (np.ndarray, float): contains the starting configurations
                                    for all Trotter replicas
        @nbs (np.ndarray, float): 3D array whose 1st dimension indexes
                                  each spin, 2nd dimension indexes
                                  neighbors to some spin, and 3rd
                                  dimension indexes the spin index
                                  of that neighbor (first element)
                                  or the coupling value to that
                                  neighbor (second element). See
                                  tools.GenerateNeighbors().
        @rng (np.RandomState): numpy random number generator object

    Returns:
        None: spins are flipped in-place within @svec
    """
    # Define some variables
    cdef int maxnb = nbs[0].shape[0]
    cdef float field = 0.0
    cdef float jperp = 0.0
    cdef int step = 0
    cdef int sidx = 0
    cdef int islice = 0
    cdef int s_nn = 0
    cdef int spinidx = 0
    cdef float jval = 0.0
    cdef float ediff = 0.0
    cdef int tleft = 0
    cdef int tright = 0

    cdef np.ndarray[np.int_t, ndim=1, negative_indices=False, mode='c'] sidx_shuff = rng.permutation(range(nspins))

    # Loop over temperatures
    for field in tqdm(sched):
        # Calculate the J_perp
        j_perp = -1 * ((slices * temp) / 2) * clog(ctanh(field / (slices * temp)))
        for step in range(mcsteps):
            # Do some number of Monte Carlo steps
            for islice in range(slices):
                # Loop over spins
                # to_be_flipped = np.empty(0, dtype=np.int64)
                sidx_shuff = rng.permutation(range(nspins))
                for sidx in sidx_shuff:
                    # Loop through the given spin's neighbors
                    for s_nn in range(maxnb):
                        # Get the neighbor spin index
                        spinidx = int(nbs[sidx, s_nn, 0])
                        # Get the coupling value to that neighbor
                        jval = nbs[sidx, s_nn, 1]
                        # Calculate the energy diff of flipping this spin
                        ediff -= 2.0 * confs[islice, sidx] * jval * confs[islice, spinidx]
                    # Periodic boundary conditions
                    if islice == 0:
                        tleft = slices - 1
                        tright = 1
                    elif islice == slices - 1:
                        tleft = slices - 2
                        tright = 0
                    else:
                        tleft = islice - 1
                        tright = islice + 1

                    ediff -= 2.0 * j_perp * confs[tleft, sidx] * confs[islice, sidx]
                    ediff -= 2.0 * j_perp * confs[islice, sidx] * confs[tright, sidx]

                    # Metropolis accept or reject
                    if ediff >= 0.0:
                        confs[islice, sidx] *= -1

                    elif cexp(ediff / (slices * temp)) > crand()/ float(RAND_MAX):
                        confs[islice, sidx] *= -1

                    ediff = 0.0

            # Perform a global move
            sidx_shuff = rng.permutation(range(nspins))
            for sidx in sidx_shuff:
                for islice in range(slices):
                    # loop through the neighbors
                    for s_nn in range(maxnb):
                        # get the neighbor spin index
                        spinidx = int(nbs[sidx, s_nn, 0])
                        # get the coupling value to that neighbor
                        jval = nbs[sidx, s_nn, 1]
                        # Calculate the energy diff of flipping this spin
                        ediff -= 2.0 * confs[islice, sidx] * jval * confs[islice, spinidx]
                # Metropolis accept or reject
                if ediff >= 0.0:
                    for trotter_i in range(slices):
                        confs[trotter_i, sidx] *= -1
                elif cexp(ediff / (slices * temp)) > crand()/float(RAND_MAX):
                    for trotter_i in range(slices):
                        confs[trotter_i, sidx] *= -1
                ediff = 0.0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
@cython.cdivision(True)
cpdef QuantumAnnealFullyConnected(np.float_t[:] sched,
                    int mcsteps,
                    int slices,
                    float temp,
                    int nspins,
                    np.float_t[:, :] confs,
                    np.float_t[:, :] couplings,
                    rng):
    """
    Adapted from Hadayat Seddiqi's code see: https://github.com/hadsed/pathintegral-qmc/

    Perform quantum annealing using path-integral quantum Monte Carlo.

    Args:
        @sched (np.array, float): an array of temperatures that specify
                                  the annealing schedule
        @mcsteps (int): number of sweeps to do on each annealing step
        @slices (int): number of replicas
        @temp (float): temperature after pre-annealing
        @temp (float): ambient temperature
        @confs (np.ndarray, float): contains the starting configurations
                                    for all Trotter replicas
        @couplings (np.ndarray, float): 2D array for the couplings between spins.
        @rng (np.RandomState): numpy random number generator object

    Returns:
        None: spins are flipped in-place within @svec
    """
    # Define some variables
    cdef float field = 0.0
    cdef float jperp = 0.0
    cdef int step = 0
    cdef int sidx = 0
    cdef int islice = 0
    cdef int s_nn = 0
    cdef float jval = 0.0
    cdef float ediff = 0.0
    cdef int tleft = 0
    cdef int tright = 0

    cdef np.ndarray[np.int_t, ndim=1, negative_indices=False, mode='c'] sidx_shuff = rng.permutation(range(nspins))

    # Loop over temperatures
    for field in tqdm(sched):
        # Calculate the J_perp
        j_perp = -1 * ((slices * temp) / 2) * clog(ctanh(field / (slices * temp)))


        for step in range(mcsteps):
            # Do some number of Monte Carlo sweeps
            for islice in range(slices):
                # Loop over spins
                sidx_shuff = rng.permutation(range(nspins))
                for sidx in sidx_shuff:
                    # Loop through the given spin's neighbors
                    for s_nn in range(nspins):
                        jval = couplings[sidx, s_nn]
                        # Calculate the energy diff of flipping this spin
                        ediff -= 2.0 * confs[islice, sidx] * jval * confs[islice, s_nn]
                    # Periodic boundary conditions
                    if islice == 0:
                        tleft = slices - 1
                        tright = 1
                    elif islice == slices - 1:
                        tleft = slices - 2
                        tright = 0
                    else:
                        tleft = islice - 1
                        tright = islice + 1

                    ediff -= 2.0 * j_perp * confs[tleft, sidx] * confs[islice, sidx]
                    ediff -= 2.0 * j_perp * confs[islice, sidx] * confs[tright, sidx]

                    # Metropolis accept or reject
                    if ediff >= 0.0:
                        confs[islice, sidx] *= -1

                    elif cexp(ediff / (slices * temp)) > crand()/ float(RAND_MAX):
                        confs[islice, sidx] *= -1

                    ediff = 0.0

            # Perform a global move
            sidx_shuff = rng.permutation(range(nspins))
            for sidx in sidx_shuff:
                for islice in range(slices):
                    # loop through all the spins
                    for s_nn in range(nspins):
                        jval = couplings[sidx, s_nn]
                        # Calculate the energy diff of flipping this spin
                        ediff -= 2.0 * confs[islice, sidx] * jval * confs[islice, s_nn]
                # Metropolis accept or reject
                if ediff >= 0.0:
                    for trotter_i in range(slices):
                        confs[trotter_i, sidx] *= -1
                elif cexp(ediff / (slices * temp)) > crand()/float(RAND_MAX):
                    for trotter_i in range(slices):
                        confs[trotter_i, sidx] *= -1
                ediff = 0.0
