## Path Integral Quantm Monte Carlo

This code is an adaptation from [Hadayat Seddiqi's code](https://github.com/hadsed/pathintegral-qmc/)

This code consists of 4 parts:

1. High performance SA and PIQMC QA Cython code. (`src/qmc.pyx` and `src/sa.pyx`)
2. A python interface to call this code (`python_interface.py`)
3. A file with different spin models, for now only the Edwards-Anderson model (`models.py`)
4. A single script to run different experiments with (`run_experiment.py`)

In order to call the Cython code, we need to install the package. First, install the packages in `environment.yml` and then run

```bash
python setup.py install
```
to install the `piqmc` module.

To test multiple experimental setups use the command line interface
```bash
python run_experiment --<ARG1>=<VALUE1> --<ARG2>=<VALUE2>  ...
```

We can save the residual energies, the MC times tested and the experiement parameters in the `./results/` folder. Figures
are saved in `./results/figures/`

We will list the availabe arguments below. The default settings are according to 
[Santoro (2002)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.66.094203)

### Quantum Annealing Parameters

**tau_schedule**: The spacing of the quantum annealing schedule. Default is [10, 100, 1000, 10000]

**mcsteps**: The number of MC steps to perform at each point in the schedule. The total number of MC steps is then equal to tau * mcsteps. 
Default value is 3.0.

**gamma_0**: The initial value of the transverse field gamma. Default value is 3.0.

**gamma_T**: The final value of the transverse field gamma. Default is 1e-8.

**P**: The number of Trotter slices. Default is 40.

**PT**: According to Santoro, the value of P*T is essential for the dynamics of the annealing. Here, T is the temperature, which is 
calculated from the value of PT. Default is 1.0.

**q_scheds**: This is a list of numpy arrays corresponding to the annealing schedules.
Per default, these arrays are linearly spaced, beginning at **gamma_0** and ending with **gamma_T** in 
**tau_schedule[i]** steps. One can in principle submit any list of schedules here, as long as they're arrays of length **tau_schedule[i]**

### Pre-Annealing Parameters

In order to make sure the system is sufficiently thermalized, we perform short thermal annealing scheme from T_0 to T (T defined above).

**preannealing_temperature**: Starting temperature for the pre-annealing schedule. Default is 3.0.

**preannealing_schedule_steps**: Number of steps in the pre-annealing schedule. Default is 60.

**preannealing_mcsteps**: Number of MC steps per point in the annealing schedule. Default is 100.

**preannealing_sched**: Linearly spaced annealing schedule starting at **preannealing_temperature** and ending at **PT / P**

### Miscaleneous

**seed**: Random seed to control the initialization and MCMC random number generator. Default is None.