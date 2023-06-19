import yaml

pr: dict = {}

with open('parameters.yml', 'r') as stream:
    try:
        pr = yaml.load(stream, Loader=yaml.FullLoader)
    except yaml.YAMLError as exc:
        print('Error parsing YAML parameters file.')
