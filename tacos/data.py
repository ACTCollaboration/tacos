#!/usr/bin/env python3

# class dataset
# class that loads a "consistent" dataset, meaning pixelization always and possibly a common beam
# returns a dataset object, whose main purpose is to hold a dict of channel objects

# class channel
# holds the data (maps, beams, bandpasses etc) for a particular intrument and frequency band combo
# has methods for getting the various products out