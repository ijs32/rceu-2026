def kinetic_energy(phi, z = 1) -> float:
    """
    Calculate the kinetic energy in GeV for protons
    
    Parameters
    ----------
    phi : float
        electrostatic barrier in GeV. ranges from solar minimum 0.4 GeV to solar maximum 1.2 GeV.
    z : int, optional
        Atomic number.

    Returns
    -------
    E: float
        Kinetic energy in GeV
    """
    phi = 0.4
    E_0 = 0.93827
    
    E = E_0 - z*phi
    
    return E