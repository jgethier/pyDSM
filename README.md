## **pyDSM**. Stress Relaxation in Entangled Polymers using the Discrete Slip-link Model (DSM) with GPU in Python


**Reference:** Ethier, Jeffrey G.; Cordoba, Andres; Schieber, Jay D. "pyDSM: Fast Quantitative Rheology Predictions for Entangled Polymers in Python", Computer Physics Communications, submitted.


<br>

---

<br>

### Installation:

<br>

(1) Create anaconda environment with dependencies:

```
conda create env --file environment.yml
```

(2) Activate conda environment:

```
conda activate pydsm-env
```

<br>


To run a simulation:

(1) Set input parameters in input.yaml

(2) Run the program
```
python gpu_dsm.py
```

POSITIONAL ARGUMENTS:

sim_ID - An integer for the simulation ID. Appended to the filenames. 

Example: 
```
python gpu_dsm.py 1
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

If the --fit flag is not used, G(t) fits can be done by importing the class in a new Python file:
```
from core.fit import CURVE_FIT

filepath='path/to/Gt_result.txt'
output_path = 'path/to/fit/results/'
gt_fit = CURVE_FIT(filepath,output_path)
gt_fit.fit()
```
