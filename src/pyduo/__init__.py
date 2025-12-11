"""
PyDuo - Python/CUDA implementation of Duo
A program for solving the rovibronic Schrödinger equation for diatomic molecules

This is a GPU-accelerated reimplementation of the Duo Fortran program
using CuPy for CUDA operations.
"""

__version__ = "0.1.0"
__author__ = "PyDuo Contributors"

from .duo import DuoSolver
from .constants import PhysicalConstants
from .grid import Grid
from .fields import PotentialField, SpinOrbitField, CouplingField
from .hamiltonian import HamiltonianBuilder
from .solver import EigenSolver

__all__ = [
    'DuoSolver',
    'PhysicalConstants',
    'Grid',
    'PotentialField',
    'SpinOrbitField',
    'CouplingField',
    'HamiltonianBuilder',
    'EigenSolver'
]
