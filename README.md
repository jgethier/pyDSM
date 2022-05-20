# pyDSM

Discrete Slip-link Model (DSM) for GPU in Python

To run a simulation:
```
Create and activate a python environment that runs python 3.7 or greater.

Install: numpy, math, pyyaml, numba, alive-progress <-pip
Install: cudatoolkits <-conda

1) Set input parameters in input.yaml

2) Run the program
>> python gpu_dsm.py num

and replace 'num' with any integer (used for random seed generation and file name)
```
