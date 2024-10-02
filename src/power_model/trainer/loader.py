import yaml


def load_pipeline(file):
    """Load the pipeline configuration from a YAML file."""

    with open(file, "r") as f:
        config = yaml.safe_load(f)

    return config
