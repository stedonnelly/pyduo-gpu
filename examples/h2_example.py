"""
Example: Computing H2 rovibronic levels with PyDuo

This example demonstrates how to set up a simple H2 calculation
using the Morse potential and compute rovibronic energy levels.
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from pyduo import DuoSolver
from pyduo.fields import PotentialField, ElectronicState
from pyduo.constants import PhysicalConstants


def main():
    print("="*60)
    print("PyDuo Example: H2 Rovibronic Levels")
    print("="*60)

    # Check for GPU
    try:
        import cupy as cp
        use_gpu = True
        print("\nGPU acceleration: ENABLED (CuPy detected)")
    except ImportError:
        use_gpu = False
        print("\nGPU acceleration: DISABLED (install CuPy for GPU support)")

    # Create solver
    solver = DuoSolver(use_gpu=use_gpu, verbose=4)

    # Define H2 ground state X 1Sigma+g
    # Using Morse potential: V(r) = De * (1 - exp(-a*(r-re)))^2
    # Parameters from literature:
    # De = 38297.7 cm^-1
    # re = 0.7414 Angstrom
    # a = 1.9426 Angstrom^-1 (from omega_e = 4401.21 cm^-1)

    Te = 0.0  # Electronic energy
    De = 38297.7  # Dissociation energy (cm^-1)
    re = 0.7414  # Equilibrium distance (Angstrom)
    omega_e = 4401.21  # Harmonic frequency (cm^-1)

    # Calculate Morse parameter from omega_e
    # omega_e = a * sqrt(2 * De / mu) * (hbar * c)
    # In appropriate units: a = omega_e * sqrt(mu / (2 * De * aston))
    mu = 0.50391  # H2 reduced mass in amu
    a = omega_e * np.sqrt(mu / (2 * De * PhysicalConstants.ASTON))

    print(f"\nH2 Morse potential parameters:")
    print(f"  Te = {Te:.2f} cm^-1")
    print(f"  De = {De:.2f} cm^-1")
    print(f"  re = {re:.4f} Angstrom")
    print(f"  a  = {a:.4f} Angstrom^-1")

    # Create electronic state
    state_X = ElectronicState(
        name="X_1Sigma+g",
        istate=0,
        lambda_=0,      # Sigma state
        multiplicity=1,  # Singlet
        spin=0.0,
        parity_pm=+1,   # Sigma+
        parity_gu=+1    # g (gerade)
    )

    # Create potential
    poten_X = PotentialField(
        name="X_1Sigma+g",
        istate=0,
        lambda_=0,
        multiplicity=1,
        func_type="MORSE",
        parameters=np.array([Te, De, re, a])
    )

    # Set up calculation
    # Grid: 500 points from 0.3 to 5.0 Angstrom
    solver.setup_manual(
        npoints=500,
        rmin=0.3,
        rmax=5.0,
        mass1=1.00782503207,  # H atom mass
        mass2=1.00782503207,
        potentials=[poten_X],
        states=[state_X],
        j_list=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],  # J = 0 to 5
        grid_type=0  # Uniform grid
    )

    print(f"\nCalculation setup:")
    print(f"  Grid: {solver.grid.npoints} points")
    print(f"  Range: [{solver.grid.rmin:.2f}, {solver.grid.rmax:.2f}] Angstrom")
    print(f"  Reduced mass: {solver.reduced_mass:.6f} amu")
    print(f"  J values: {solver.j_list}")

    # Run calculation
    print("\nRunning calculation...")
    solver.run()

    # Print results
    solver.print_summary()

    # Print detailed vibrational levels
    print("\n" + "-"*60)
    print("Vibrational Energy Levels (J=0)")
    print("-"*60)

    if 0 in solver.vibrational_energies:
        vib_e = solver.vibrational_energies[0]
        zpe = vib_e[0]
        print(f"  ZPE = {zpe:.4f} cm^-1")
        print(f"\n  {'v':>4s}  {'E (cm^-1)':>14s}  {'E-ZPE (cm^-1)':>14s}  {'E (eV)':>12s}")
        print("  " + "-"*50)

        for v in range(min(15, len(vib_e))):
            e = vib_e[v]
            e_rel = e - zpe
            e_ev = e / PhysicalConstants.EV
            print(f"  {v:4d}  {e:14.4f}  {e_rel:14.4f}  {e_ev:12.6f}")

    # Print rovibrational levels for each J
    print("\n" + "-"*60)
    print("Rovibronic Energy Levels")
    print("-"*60)

    # Get ground state energy (J=0, v=0)
    if 0.0 in solver.rovibronic_energies:
        e0 = solver.rovibronic_energies[0.0][0]
    else:
        e0 = 0.0

    print(f"\n  {'J':>4s}  {'i':>4s}  {'E (cm^-1)':>14s}  {'E-E0 (cm^-1)':>14s}")
    print("  " + "-"*45)

    for J in sorted(solver.rovibronic_energies.keys()):
        energies = solver.rovibronic_energies[J]
        for i, e in enumerate(energies[:5]):  # First 5 levels per J
            e_rel = e - e0
            print(f"  {J:4.1f}  {i:4d}  {e:14.4f}  {e_rel:14.4f}")
        print()

    # Compare with expected values
    print("\n" + "-"*60)
    print("Comparison with Literature")
    print("-"*60)

    # Expected values for H2 X 1Sigma+g (from NIST)
    expected = {
        'ZPE': 2179.3,      # Experimental ZPE
        'omega_e': 4401.21,  # Harmonic frequency
        'omega_e_xe': 121.33, # Anharmonicity
    }

    if 0 in solver.vibrational_energies:
        vib_e = solver.vibrational_energies[0]
        calc_zpe = vib_e[0]
        if len(vib_e) >= 3:
            # Estimate omega_e and omega_e_xe from first 3 levels
            G0 = vib_e[0]
            G1 = vib_e[1]
            G2 = vib_e[2]
            delta_10 = G1 - G0
            delta_21 = G2 - G1

            calc_omega_e = 0.5 * (3 * delta_10 - delta_21)
            calc_omega_e_xe = 0.25 * (delta_10 - delta_21)

            print(f"\n  {'Quantity':20s}  {'Calculated':>14s}  {'Literature':>14s}")
            print("  " + "-"*55)
            print(f"  {'ZPE (cm^-1)':20s}  {calc_zpe:14.4f}  {expected['ZPE']:14.4f}")
            print(f"  {'omega_e (cm^-1)':20s}  {calc_omega_e:14.4f}  {expected['omega_e']:14.4f}")
            print(f"  {'omega_e*xe (cm^-1)':20s}  {calc_omega_e_xe:14.4f}  {expected['omega_e_xe']:14.4f}")

    print("\nCalculation complete!")


if __name__ == "__main__":
    main()
