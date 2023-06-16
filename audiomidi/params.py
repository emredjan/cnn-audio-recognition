from pathlib import Path

import yaml

with open('parameters.yml', 'r') as stream:
    try:
        p = yaml.load(stream, Loader=yaml.FullLoader)
    except yaml.YAMLError as exc:
        print('Error parsing YAML parameters file.')
        p = {}

# file locations
fl = p['file_locations']
nsynth_train_audio = Path(fl['nsynth_train_audio'])
nsynth_test_audio = Path(fl['nsynth_test_audio'])
nsynth_valid_audio = Path(fl['nsynth_valid_audio'])
features_dir = Path(fl['features_dir'])

train_data_path = Path(fl['train_data_path'])
valid_data_path = Path(fl['valid_data_path'])
test_data_path = Path(fl['test_data_path'])
train_additional_data_path = Path(fl['train_additional_data_path'])
valid_additional_data_path = Path(fl['valid_additional_data_path'])
test_additional_data_path = Path(fl['test_additional_data_path'])
train_names_path = Path(fl['train_names_path'])
valid_names_path = Path(fl['valid_names_path'])
test_names_path = Path(fl['test_names_path'])
train_metadata_path = Path(fl['train_metadata_path'])
valid_metadata_path = Path(fl['valid_metadata_path'])
test_metadata_path = Path(fl['test_metadata_path'])

weights_dir = Path(fl['weights_dir'])
log_dir = Path(fl['log_dir'])

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

# training_parameters:
tp = p['training_parameters']
epochs = tp['epochs']
batch_size = tp['batch_size']
early_stop = tp['early_stop']
weight_file_pattern = tp['weight_file_pattern']
