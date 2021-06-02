import numpy as np
from scipy.interpolate import interp1d
import os 

from pixell import enmap
import h5py

# class is overkill, just need 3 load functions that return array