import argparse
import numpy as np
import h5py
import sys







def get_values():

    with h5py.File("NuLib_SFHo.h5", "r") as f:

        E=np.array(f["neutrino energies"])
        rho=np.array(f["rho_ppoints"])
        T=np.array(f["temp_points"])
        Y=np.array(f["ye_points"])