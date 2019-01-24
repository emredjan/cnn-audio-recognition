from pathlib import Path

import yaml

with open('parameters.yml', 'r') as stream:
    try:
        p = yaml.load(stream)
    except yaml.YAMLError as exc:
        print('Error parsing YAML parameters file.')

# file locations
fl = p['file_locations']

# nsynth
nsynth = p['nsynth']
nsynth_sr = nsynth['sample_rate']
