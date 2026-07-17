## README File
Derivations for q1-4 were derived symbolically in Python using the SymPy library. For q5, a 1D FVM solver was written in Python for the compressible Euler equations using Stager-Warming (sw) and Van Leer (vl) flux splitting schemes. The NumPy and Pandas libraries were used for computation and data handling.

### File Structure
```
solver.py        # Contains Physics and Solver classes
main.py          # Runs a simulation using a chosen scheme
Run              # Bash script that prompts for scheme and runs Python
sw_results.csv   # Contains results for test case 1 using Stager-warming flux splitting 
vl_results.csv   # Contains results for test case 1 using Van Leer flux splitting
1_derive.ipynb   # Contains derivations for q1-4 using Sympy library.]
2_plot.ipynb     # Jupyter Notebook for running solvers and plotting results 
```
### Dependencies 
The solver was developed using the following versions: 
- Python 3.13.2
- NumPy 2.4.2
- Pandas 3.0.1
SymPy 1.14.0 and Matplotlib 3.10.8 was used for derivation and visualization only and are not required for running the basic solver. 

### Using the Solver Class
The Solver class provides an interface for running 1D FVM compressible solver.  To use it, you need to specify:
- flux scheme ("sw" or "vl"),
- domain (xmin, xmax),
- number of divisions (N),
- left and right initial conditions (wl, wr). 
Optional parameters include gamma, CFL, and Umax.
#### Test Case 1
For test case 1 the following values are used. 
``` 
Solver(
    scheme="sw",
    xmin=-10,
    xmax=10,
    N=50,
    wl={"rho": 1.0, "u": 0.0, "p": 100000.0},
    wr={"rho": 0.125, "u": 0.0, "p": 10000.0},
    CFL=0.4
)
``` 
The scheme can be either ``sw`` (Stager-Warming) or ``vl`` (Van Leer).
### How to run the solver
1. Install dependencies  ``pip install numpy pandas``
2. Run the script using bashscript ``./Run`` or directly with ``python3 main.py``
3. Enter flux scheme (sw/vl)
4. Results are output to a csv file, which includes columns x, density, velocity, total_energy, pressure, speed_of_sound, and mach. This can be opened in excel. 



