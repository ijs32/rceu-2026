def local_interstellar_spectrum(E):
    """
    Calculate the Local Interstellar Spectrum (LIS) for protons
    using a standard empirical fit (Vos & Potgieter parameterization proxy).
    
    E: Kinetic energy in GeV
    Returns: Intensity J in particles / (m^2 * sr * s * GeV)
    """
    # case of proton LIS
    A = 1.9e4
    B = 0.67
    return (A * E**1.01) / (E + B)**2.7