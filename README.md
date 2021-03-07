# Simulated Classical and Quantum Annealing

[![License](https://img.shields.io/badge/License-Do%20No%20Harm-blue)](https://github.com/therooler/piqmc/blob/master/LICENSE.md)

This repository contains an implementation of simulated annealing (SA) and simulated quantum annealing (SQA) with path integral monte carlo (PIQMC). The interface to the code is written in Python 3 and the Monte Carlo sampling is written in Cython. This code was used to produce some of the results of this paper https://arxiv.org/abs/2101.10154.

Here, we have an implementation that supports the 2D Edwards-Anderson model, the Sherrington-Kirkpatrick model and the Wishart Planted Ensemble.

We note that this code is an adaptation from [Hadayat Seddiqi's Cython code](https://github.com/hadsed/pathintegral-qmc/). We fixed some bugs, added global moves to the quantum annealing monte carlo dynamics and stripped the code of things that were not needed for our purposes.

## Installation
In order to call the Cython code, we need to install the package. First, install the packages in `environment.yml` and then run

```bash
python setup.py install
```
to install the `piqmc` module. The installation takes less than one minute with a typical CPU if the the packages in `environment.yml` are already available.

To test multiple experimental setups use the command line interface
```bash
python run_<EXPERIMENT>.py --<ARG1>=<VALUE1> --<ARG2>=<VALUE2>  ...
```
See the respective files to see which arguments can be passed.
We can save the residual energies, the MC times tested and the experiement parameters in the `./results/` folder.

## Content

This implementation consists of 4 parts:

* High performance SA and PIQMC QA Cython code. (`src/qmc.pyx` and `src/sa.pyx`)
* A python interface to call this code (`python_interface.py`) that contains a `QuantumPIAnneal` class for PIQMC and `ClassicalAnneal` class for SA.
* A file with different spin models. Our implementation supports the 2D Edwards-Anderson model, and fully-connected models such as the Sherrington-Kirkpatrick model and the Wishart Planted Ensemble (`models.py`).
* Scripts to run different annealing experiments for either SA with (`run_SA_....py`) or for PIQMC with (`run_PIQMC_....py`). Here, each run file corresponds to a different model.

We will list the availabe arguments below. The default settings are similar to 
[Santoro (2002)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.66.094203).

### `Quantum Annealing Parameters`

**tau_schedule**: The spacing of the quantum annealing schedule. Default is [1, 10, 100, 1000,10000,100000].

**mcsteps**: The number of MC sweeps to perform at each point in the schedule. The total number of MC steps is then equal to tau * mcsteps. 
Default value is 1.0.

**gamma_0**: The initial value of the transverse field gamma. Default value is 1.0.

**gamma_T**: The final value of the transverse field gamma. Default is 1e-8.

**P**: The number of Trotter slices. Default is 20.

**PT**: According to Santoro, the value of P*T is essential for the dynamics of the annealing. Here, T is the temperature, which is 
calculated from the value of PT. Default is 1.0.

**q_scheds**: This is a list of numpy arrays corresponding to the annealing schedules.
Per default, these arrays are linearly spaced, beginning at **gamma_0** and ending with **gamma_T** in 
**tau_schedule[i]** steps. One can in principle submit any list of schedules here, as long as they're arrays of length **tau_schedule[i]**

#### `Pre-Annealing Parameters`

In order to make sure the system is sufficiently thermalized, we perform short thermal annealing scheme from T_0 to T (T defined above).

**preannealing_temperature**: Starting temperature for the pre-annealing schedule. Default is 3.0.

**preannealing_schedule_steps**: Number of steps in the pre-annealing schedule. Default is 60.

**preannealing_mcsteps**: Number of MC steps per point in the annealing schedule. Default is 100.

**preannealing_sched**: Linearly spaced annealing schedule starting at **preannealing_temperature** and ending at **PT / P**

### `Classical Annealing Parameters`

**tau_schedule**: The spacing of the classical annealing schedule. Default is [1, 10, 100, 1000,10000,100000].

**mcsteps**: The number of MC sweeps to perform at each point in the schedule. The total number of MC steps is then equal to tau * mcsteps. 
Default value is 1.0.

**T_0**: The initial value of temperature. Default value is 1.0.

**T_f**: The final value of the temperature. Default is 1e-8.

**num_warmup**: The number of warmup steps to thermalize SA at T_0.

**T_scheds**: This is a list of numpy arrays corresponding to the annealing schedules.
Per default, these arrays are linearly spaced, beginning at **T_0** and ending with **T_f** in 
**tau_schedule[i]** steps. One can in principle submit any list of schedules here, as long as they are arrays of length **tau_schedule[i]**

### `Miscellaneous`

**numruns**: Number of SA/PIQMC annealing runs. Each run corresponds to a specific random seed to control the initialization and MCMC random number generator.

**seed**: Random seed to identify the random instance of couplings to be imported from the `data` folder.

## Speed illustration of our code

Using an `Intel(R) Xeon(R) CPU E5-2683 v4 @ 2.10GHz`, the typical number of monte carlo steps for PIQMC with 20 trotter slices on the 2D Edwards-Anderson model with 40x40 spins is ~50 per second. For SA, ~2000 monte carlo steps per second are performed on the same model. Similarly for the Sherrington-Kirkpatrick model with 100 spins, we have ~50 iterations per second for PIQMC with 100 trotter slices, while ~9000 iteractions per second for SA.

## License

This code is licensed under the [Do No Harm](https://github.com/therooler/piqmc/blob/master/LICENSE.md) license and is intended for academic research that will be beneficial to humanity.

## Citing
```bibtex
@misc{VNA2021,
      title={Variational Neural Annealing}, 
      author={Mohamed Hibat-Allah and Estelle M. Inack and Roeland Wiersema and Roger G. Melko and Juan Carrasquilla},
      year={2021},
      eprint={2101.10154},
      archivePrefix={arXiv},
      primaryClass={cond-mat.dis-nn}
}
```
