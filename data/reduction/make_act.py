#!/usr/bin/envs python3

# a script that only runs one and contains all the messiness to go from the raw
# data in tacos/raw to the reduces data.
#
# specific to act data. although act ivar maps have IQU covar, we coadd just using diagonal

from pixell import enmap
from tacos import data

import re

# first do the maps
# all coaddition occurs in pixel space
mapsets = dict(
map_f90_set0_fns = ['map_pa5_f090_night_set0.fits', 'map_pa6_f090_night_set0.fits'],
map_f90_set1_fns = ['map_pa5_f090_night_set1.fits', 'map_pa6_f090_night_set1.fits'],
map_f150_set0_fns = ['map_pa4_f150_night_set0.fits', 'map_pa5_f150_night_set0.fits', 'map_pa6_f150_night_set0.fits'],
map_f150_set1_fns = ['map_pa4_f150_night_set1.fits', 'map_pa5_f150_night_set1.fits', 'map_pa6_f150_night_set1.fits'],
map_f220_set0_fns = ['map_pa4_f090_night_set0.fits'],
map_f220_set1_fns = ['map_pa4_f090_night_set1.fits'],
)

mapsets.update(dict(
    map_f90_fns = mapsets['map_f90_set0_fns'] + mapsets['map_f90_set1_fns'],
    map_f150_fns = mapsets['map_f150_set0_fns'] + mapsets['map_f150_set1_fns'],
    map_f220_fns = mapsets['map_f220_set0_fns'] + mapsets['map_f220_set1_fns'],
))

ivarsets = {}
for k, v in mapsets.items():
    ivark = 'ivar' + k.split('map')[1]
    ivarv = []
    for mapfn in v:
        ivarv.append('div' + mapfn.split('map')[1])
    ivarsets[ivark] = ivarv

# get coadded maps
