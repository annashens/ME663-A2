from solver import Solver
import sys

# Initial Conditions: split into two regions by initial discontinuity
WL = {"rho":1.0,   "u":0.0, "p":100000.0}
WR = {"rho":0.125, "u":0.0, "p":10000.0}

# Boundary conditions: Fixed (Dirichlet) at both ends
XMIN = -10.0
XMAX = 10.0
N = 50
CFL = 0.4
GAMMA = 1.4 # air
T_END = 0.01

# Get scheme
if len(sys.argv) > 1:
    # Read scheme from command line
    SCHEME = sys.argv[1].lower()
else:
    SCHEME = input("Enter flux scheme (sw or vl): ").strip().lower()

if SCHEME not in ("sw", "vl"):
    raise ValueError("Invalid scheme. Please select 'sw' or 'vl'.")

solver = Solver(
    scheme=SCHEME,
    xmin=XMIN,
    xmax=XMAX,
    N=N,
    wl=WL,
    wr=WR,
    CFL=CFL
)

results_df = solver.run(0.0, T_END)
results_df.to_csv(f"{SCHEME}_results.csv", index=False)

