"""
Physical constants and numerical precision settings for PyDuo

Based on CODATA 2010 values, matching the original Duo Fortran implementation.
"""

import numpy as np

class PhysicalConstants:
    """Physical constants used throughout PyDuo calculations."""

    # Precision
    DTYPE = np.float64
    SMALL = np.finfo(np.float64).eps
    SAFE_MAX = np.exp(np.log(np.finfo(np.float64).max) * 0.25)
    SAFE_MIN = np.exp(np.log(np.finfo(np.float64).tiny) * 0.25)

    # Mathematical constants
    PI = np.pi
    TWOPI = 2.0 * np.pi
    SQRT2 = np.sqrt(2.0)
    SQRT3 = np.sqrt(3.0)
    RAD = 180.0 / np.pi

    # Physical constants (CODATA 2010)
    PLANCK = 6.62606957e-27        # Planck constant in erg*second
    VELLGT = 2.99792458e+10        # Speed of light in cm/second
    AVOGNO = 6.02214129e+23        # Avogadro constant
    BOLTZ = 1.3806488e-16          # Boltzmann constant in erg/Kelvin
    BOHR = 0.52917721092           # Bohr radius in Angstroms
    HARTREE = 219474.6313708       # Hartree in cm^-1
    UMA = 1.660538921e-24          # Unified atomic mass unit in grams
    E_CHARGE = 1.602176565e-19     # Electron charge in Coulombs
    G_S = 2.00231930436256         # Spin Landé g-factor

    # Derived constants
    ASTON = PLANCK / (8.0 * PI**2 * VELLGT * UMA * 1e-16)  # Rotational factor in cm^-1 amu Ang^2
    UMATOAU = BOHR**2 * HARTREE / (2.0 * ASTON)            # uma to a.u. (electron masses)
    ME = UMA / UMATOAU                                      # Electron mass in grams
    ALPHA = 2.0 * PI * BOHR * HARTREE * 1e-8               # Fine structure constant
    TODEBYE = E_CHARGE * PLANCK * 1e17 / (2.0 * PI * ME * ALPHA)  # 1 a.u. of dipole in debye
    EV = 1e7 * E_CHARGE / (PLANCK * VELLGT)                # eV in cm^-1
    KJOULE = 1e10 / (PLANCK * VELLGT * AVOGNO)             # 1 kJ/mol in cm^-1
    KCAL = KJOULE * 4.184                                   # 1 Cal in cm^-1
    THZ = 1e12 / VELLGT                                     # THz in cm^-1
    HC = VELLGT * PLANCK                                    # h*c

    @classmethod
    def print_constants(cls):
        """Print all physical constants used."""
        print("Physical constants used by PyDuo:")
        print(f"{'Planck constant h':40s} = {cls.PLANCK:20.12e} erg*second")
        print(f"{'Speed of light c':40s} = {cls.VELLGT:20.4f} cm/second")
        print(f"{'Bohr radius a0':40s} = {cls.BOHR:20.13f} angstroms")
        print(f"{'Hartree energy Eh':40s} = {cls.HARTREE:20.10f} cm^-1")
        print(f"{'Unified atomic mass unit u':40s} = {cls.UMA:20.12e} grams")
        print(f"{'Unified atomic mass unit u':40s} = {cls.UMATOAU:20.13f} me")
        print(f"{'Electron charge e':40s} = {cls.E_CHARGE:20.12e} Coulombs")
        print(f"{'Boltzmann constant kB':40s} = {cls.BOLTZ:20.12e} erg/Kelvin")
        print(f"{'Rotational factor':40s} = {cls.ASTON:20.12e} cm-1 amu Ang^2")
        print(f"{'Fine structure constant 1/alpha':40s} = {1.0/cls.ALPHA:20.14f}")
        print(f"{'1 eV':40s} = {cls.EV:20.10f} cm^-1")
        print(f"{'1 kCal/mol':40s} = {cls.KCAL:20.10f} cm^-1")
        print(f"{'1 kJ/mol':40s} = {cls.KJOULE:20.10f} cm^-1")


from functools import lru_cache


@lru_cache(maxsize=4096)
def faclog(a: float) -> float:
    """
    Calculate log(a!) for non-negative integer a.

    Parameters
    ----------
    a : float
        Non-negative number (will be rounded to integer)

    Returns
    -------
    float
        log(a!)
    """
    k = int(round(a))
    if k < 2:
        return 0.0
    v = 0.0
    for j in range(2, k + 1):
        v += np.log(float(j))
    return v


# [phase P] The solver evaluates three_j inside per-basis-pair loops --
# >100k calls that collapse to a handful of distinct argument tuples.
@lru_cache(maxsize=65536)
def three_j(a: float, b: float, c: float, al: float, be: float, ga: float) -> float:
    """
    Calculate the Wigner 3j symbol.

    Parameters
    ----------
    a, b, c : float
        Angular momenta (j1, j2, j)
    al, be, ga : float
        Projections (m1, m2, m) - note: m = -m1 - m2 for non-zero 3j

    Returns
    -------
    float
        Value of the 3j symbol

    Notes
    -----
    The 3j symbol is:
    / a   b   c  \
    \ al  be  ga /

    Selection rules:
    - Triangle inequality: |a-b| <= c <= a+b
    - Projection: al + be + ga = 0
    - |al| <= a, |be| <= b, |ga| <= c
    """
    # Selection rules
    if c > a + b:
        return 0.0
    if c < abs(a - b):
        return 0.0
    if a < 0 or b < 0 or c < 0:
        return 0.0
    if abs(al) > a or abs(be) > b or abs(ga) > c:
        return 0.0
    if abs(-ga - (al + be)) > 1e-10:  # al + be + ga = 0 => -ga = al + be
        return 0.0

    # Compute delta(abc) using log factorials
    delta_log = faclog(a + b - c) + faclog(a + c - b) + faclog(b + c - a) - faclog(a + b + c + 1.0)

    # Compute terms
    term1 = faclog(a + al) + faclog(a - al)
    term2 = faclog(b - be) + faclog(b + be)
    term3 = faclog(c + ga) + faclog(c - ga)

    termlog = (term1 + term2 + term3 + delta_log) * 0.5

    term = np.sqrt(2.0 * c + 1.0)

    # Compute summation (Brink and Satchler eq 2.34)
    newmin = int(round(max(a + be - c, b - c - al, 0.0)))
    newmax = int(round(min(a - al, b + be, a + b - c)))

    summ = 0.0
    for new in range(newmin, newmax + 1):
        dnew = float(new)
        term4 = faclog(a - al - dnew) + faclog(c - b + al + dnew)
        term5 = faclog(b + be - dnew) + faclog(c - a - be + dnew)
        term6 = faclog(dnew) + faclog(a + b - c - dnew)

        term16 = termlog - (term4 + term5 + term6)
        summ += ((-1.0) ** new) * np.exp(term16)

    # Clebsch-Gordon coefficient
    clebsh = term * summ

    # Convert Clebsch-Gordon to 3j symbol
    iphase = int(round(a - b - ga))
    minus = -1.0 if (iphase % 2) != 0 else 1.0

    return minus * clebsh / term
