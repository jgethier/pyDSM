# pyDSM

Discrete Slip-link Model (DSM) for GPU in Python

To run a simulation:
```
Create and activate a python environment that runs python 3.7 or greater.

Install: numpy, math, pyyaml, numba, alive-progress <-pip
Install: cudatoolkits <-conda

1) Set input parameters in input.yaml

2) Run the program
>> python gpu_dsm.py

```

POSITIONAL ARGUMENTS:
```
sim_ID - An integer for the simulation ID. Appended to the filenames. Example: python gpu_dsm.py 1
```

FLAG ARGUMENTS:

```
-h, --help - show help message and exit
-d [device_num] - if multiple GPUs are present, select device number
-c [otf] - force simulation to use on-the-fly (otf) correlator, but correlation errors will not be reported
-o [output_dir] - specify output directory
--fit - a flag to turn on G(t) fitting after simulation is done. 
```
