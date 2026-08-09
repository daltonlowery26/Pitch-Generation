import yaml


def load_config(name):
    path = f"config/{name}.yml"
    with open(path) as f:
        return yaml.safe_load(f)
