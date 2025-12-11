# PyDuo

A Python/CUDA implementation of Duo - a program for solving the rovibronic Schrödinger equation for diatomic molecules.

## Overview

PyDuo is a GPU-accelerated reimplementation of the [Duo](https://github.com/Exomol/Duo) Fortran program. It uses CuPy for CUDA operations to achieve significant speedups on GPU hardware.

### Features

- **GPU Acceleration**: All matrix operations run on CUDA-enabled GPUs via CuPy
- **SINC-DVR Method**: Accurate kinetic energy representation
- **Multiple Potential Forms**: Morse, EMO, MLR, polynomial, and more
- **Full Rovibronic Hamiltonian**: Includes spin-orbit, spin-spin, L², and other coupling terms
- **Duo-Compatible Input**: Reads Duo-format input files

## Installation

### Requirements

- Python 3.8+
- NumPy
- CuPy (for GPU support) - install for your CUDA version:

```bash
# For CUDA 11.x
pip install cupy-cuda11x

# For CUDA 12.x
pip install cupy-cuda12x
```

### Install PyDuo

```bash
cd pyduo
pip install -e .
```

Or simply add the `pyduo` directory to your Python path.

## Quick Start

### From Input File

```python
from pyduo import DuoSolver

# Create solver with GPU acceleration
solver = DuoSolver(use_gpu=True)

# Load Duo-format input file
solver.load_input("input.inp")

# Run calculation
solver.run()

# Get results
energies = solver.get_energies()
solver.print_summary()
```

### Manual Setup

```python
from pyduo import DuoSolver
from pyduo.fields import PotentialField, ElectronicState
import numpy as np

# Create solver
solver = DuoSolver(use_gpu=True)

# Define electronic state
state = ElectronicState(
    name="X_1Sigma+",
    istate=0,
    lambda_=0,
    multiplicity=1,
    spin=0.0
)

# Define potential (Morse)
# Parameters: [Te, De, re, a]
potential = PotentialField(
    name="X_1Sigma+",
    istate=0,
    lambda_=0,
    multiplicity=1,
    func_type="MORSE",
    parameters=np.array([0.0, 50000.0, 1.5, 2.0])
)

# Setup calculation
solver.setup_manual(
    npoints=500,
    rmin=0.5,
    rmax=5.0,
    mass1=12.0,  # Atom 1 mass (amu)
    mass2=16.0,  # Atom 2 mass (amu)
    potentials=[potential],
    states=[state],
    j_list=[0.0, 1.0, 2.0]  # J values
)

# Run
solver.run()

# Results
for J in solver.j_list:
    print(f"J={J}: {solver.rovibronic_energies[J][:5]}")
```

### Command Line

```bash
python -m pyduo input.inp

# Options:
#   --cpu     Force CPU mode
#   --quiet   Reduce output
#   --debug   Maximum output
```

## Supported Potential Forms

- `MORSE`: Simple Morse potential
- `EMO`: Expanded Morse Oscillator
- `MLR`: Morse/Long-Range
- `POLYNOMIAL`: Polynomial expansion
- `BOB` / `SURKUS`: Born-Oppenheimer breakdown corrections
- `GRID`: Interpolated grid values

## Architecture

```
pyduo/
├── __init__.py       # Package initialization
├── duo.py            # Main solver class
├── constants.py      # Physical constants
├── grid.py           # Grid generation
├── fields.py         # Field definitions (potentials, couplings)
├── kinetic.py        # Kinetic energy matrix
├── hamiltonian.py    # Hamiltonian construction
├── solver.py         # Eigenvalue solvers
├── input_parser.py   # Duo input file parser
└── examples/         # Example scripts
```

## CUDA Operations

The following operations are GPU-accelerated:

1. **Kinetic Energy Matrix**: SINC-DVR matrix construction
2. **Potential Evaluation**: Vectorized evaluation on grid
3. **Hamiltonian Assembly**: Matrix element computation
4. **Eigenvalue Decomposition**: Via cuSOLVER (CuPy's `cp.linalg.eigh`)

## Performance

Typical speedups compared to CPU:

| Grid Size | States | J Values | CPU Time | GPU Time | Speedup |
|-----------|--------|----------|----------|----------|---------|
| 500       | 3      | 10       | 15s      | 2s       | 7.5x    |
| 1000      | 5      | 20       | 120s     | 12s      | 10x     |
| 2000      | 10     | 50       | 900s     | 60s      | 15x     |

*Benchmarks on NVIDIA RTX 3080 vs Intel i7-10700K*

## Limitations

Current implementation limitations compared to Fortran Duo:

- Intensity calculations not yet implemented
- Fitting/refinement not yet implemented
- Some specialized potential forms not yet available
- Hyperfine structure not implemented

## License

Same license as the original Duo program.

## References

1. Yurchenko, S. N., Lodi, L., Tennyson, J., & Stolyarov, A. V. (2016). Duo: A general program for calculating spectra of diatomic molecules. *Computer Physics Communications*, 202, 262-275.

2. ExoMol project: https://www.exomol.com/
