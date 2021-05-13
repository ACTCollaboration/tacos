#!/usr/bin/envs python3

# a script that only runs one and contains all the messiness to go from the raw
# data in tacos/raw to the reduces data.
# specific to act data

from pixell import enmap
from tacos import data

import re

# first do the maps
# all coaddition occurs in pixel space
freqs = [90, 150, 220]
splits = [0, 1]

maps = []
ivars = []
mapstr = 'map_()_f{freq}_night_set{split}.fits'
ivarstr = 'div_()_f220_night_set0.fits'

# first load all the data
for freq in freqs:
    for split in splits:
        map_fn = mapstr.format(freq, split)
        ivar_fn = ivarstr
        maps.append(enmap.read_map(map_fn))

# do splits coadded across freqs


# do coadds across both split and freqs
