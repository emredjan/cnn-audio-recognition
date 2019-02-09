from pathlib import Path

import yaml

with open('parameters.yml', 'r') as stream:
    try:
        p = yaml.load(stream)
    except yaml.YAMLError as exc:
        print('Error parsing YAML parameters file.')

# file locations
fl = p['file_locations']
nsynth_train_audio = Path(fl['nsynth_train_audio'])
nsynth_test_audio = Path(fl['nsynth_test_audio'])
nsynth_valid_audio = Path(fl['nsynth_valid_audio'])
features_dir = Path(fl['features_dir'])

# nsynth
nsynth = p['nsynth']
nsynth_sr = nsynth['sample_rate']
nsynth_max_seconds = nsynth['max_seconds']
nsynth_train_audio = Path(nsynth_train_audio)
nsynth_test_audio = Path(nsynth_test_audio)
nsynth_valid_audio = Path(nsynth_valid_audio)

# runtime_parameters
rp = p['runtime_parameters']
librosa_spec_windows = rp['librosa_spec_windows']
librosa_hop_length = rp['librosa_hop_length']
