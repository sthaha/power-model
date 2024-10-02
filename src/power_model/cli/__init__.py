# SPDX-FileCopyrightText: 2024-present Sunil Thaha <sthaha@redhat.com>
#
# SPDX-License-Identifier: MIT
import click

from power_model.__about__ import __version__
from power_model.trainer import load_pipeline, run_pipeline


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
)
@click.version_option(version=__version__, prog_name="power-model")
def pm():
    pass


@pm.command()
@click.option(
    "-f",
    "--file",
    required=True,
    help="Path to the pipeline YAML file.",
    type=click.Path(exists=True),
)
def train(file):
    """Train models based on the provided pipeline configuration."""

    try:
        pipeline = load_pipeline(file)
        run_pipeline(pipeline)
        click.echo("Training completed successfully.")

    except Exception as e:
        click.echo(f"An error occurred: {e}")


if __name__ == "__main__":
    train()
