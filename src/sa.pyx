cimport cython
cimport numpy as np

from tqdm import tqdm
from libc.math cimport exp as cexp
from libc.stdlib cimport rand as crand
from libc.stdlib cimport RAND_MAX as RAND_MAX


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
@cython.cdivision(True)
cpdef AnnealFullyConnected(np.float_t[:] sched,
             int mcsteps,
             np.float_t[:] svec,
             np.float_t[:, :] couplings,
             rng):
    """
    Adapted from Hadayat Seddiqi's code see: https://github.com/hadsed/pathintegral-qmc/

    Execute thermal annealing according to @sched with @mcsteps
    sweeps for each annealing step. Starting configuration is
    given by @svec, which we update in-place and calculate energies
    using the couplings array that would correspond to a fully connected spin system

    Args:
        @sched (np.array, float): an array of temperatures that specify
                                  the annealing schedule
        @mcsteps (int): number of sweeps to do on each annealing step
        @svec (np.array, float): contains the starting configuration
        @couplings (np.ndarray, float): 2D array of couplings between spins.
        @rng (np.RandomState): numpy random number generator object

    Returns:
        None: spins are flipped in-place within @svec
    """
    # Define some variables
    cdef int nspins = svec.size
    cdef int schedsize = sched.size
    cdef int itemp = 0
    cdef float temp = 0.0
    cdef int step = 0
    cdef int sidx = 0
    cdef int s_nn = 0
    cdef float jval = 0.0
    cdef float ediff = 0.0
    cdef np.ndarray[np.int_t, ndim=1] sidx_shuff = rng.permutation(range(nspins))

    # Loop over temperatures
    for itemp in tqdm(range(sched.size)):
        # Get temperature
        temp = sched[itemp]
        # Do some number of Monte Carlo steps
        for step in range(mcsteps):
            # Loop over spins
            for sidx in sidx_shuff:
                # loop through all the spins
                for s_nn in range(nspins):
                    jval = couplings[sidx,s_nn]
                    ediff += -2.0 * svec[sidx] * (jval*svec[s_nn])
                # Metropolis accept or reject
                if ediff >= 0.0:  # avoid overflow
                    svec[sidx] *= -1
                elif cexp(ediff/temp) > crand()/float(RAND_MAX):
                    svec[sidx] *= -1
                # Reset energy diff value
                ediff = 0.0
            sidx_shuff = rng.permutation(sidx_shuff)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
@cython.cdivision(True)
cpdef Anneal(np.float_t[:] sched,
             int mcsteps,
             np.float_t[:] svec,
             np.float_t[:, :, :] nbs,
             rng):
    """
    Adapted from Hadayat Seddiqi's code see: https://github.com/hadsed/pathintegral-qmc/

    Execute thermal annealing according to @sched with @mcsteps
    sweeps for each annealing step. Starting configuration is
    given by @svec, which we update in-place and calculate energies
    using the "neighbors array" @nbs.

    Args:
        @sched (np.array, float): an array of temperatures that specify
                                  the annealing schedule
        @mcsteps (int): number of sweeps to do on each annealing step
        @svec (np.array, float): contains the starting configuration
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
    cdef int nspins = svec.size
    cdef int maxnb = nbs[0].shape[0]
    cdef int schedsize = sched.size
    cdef int itemp = 0
    cdef float temp = 0.0
    cdef int step = 0
    cdef int sidx = 0
    cdef int si = 0
    cdef int spinidx = 0
    cdef float jval = 0.0
    cdef float ediff = 0.0
    cdef np.ndarray[np.int_t, ndim=1] sidx_shuff = rng.permutation(range(nspins))

    # Loop over temperatures
    for itemp in tqdm(range(sched.size)):
        # Get temperature
        temp = sched[itemp]
        # Do some number of Monte Carlo steps
        for step in range(mcsteps):
            # Loop over spins
            for sidx in sidx_shuff:
                # loop through the given spin's neighbors
                for si in range(maxnb):
                    # get the neighbor spin index
                    spinidx = int(nbs[sidx,si,0])
                    # get the coupling value to that neighbor
                    jval = nbs[sidx,si,1]
                    # self-connections are not quadratic
                    if spinidx == sidx:
                        ediff += -2. * svec[sidx]*jval
                    # calculate the energy diff of flipping this spin
                    else:
                        ediff += -2.0 * svec[sidx] * (jval*svec[spinidx])
                # Metropolis accept or reject
                if ediff >= 0.0:  # avoid overflow
                    svec[sidx] *= -1
                elif cexp(ediff/temp) > crand()/float(RAND_MAX):
                    svec[sidx] *= -1
                # Reset energy diff value
                ediff = 0.0
            sidx_shuff = rng.permutation(sidx_shuff)