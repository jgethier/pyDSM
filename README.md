# pyDSM

Discrete Slip-link Model (DSM) for GPU in Python

Set up a conda environment:
```
1) Install Miniconda (or Anaconda)
2) In Anaconda prompt, create new environment from file:
>> conda create env --file environment.yml
>> conda activate pydsm-env

```
To run a simulation:
```
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
--distr - a flag to save initial and final Q, Lpp, and Z distributions to file.
```
