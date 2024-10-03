# SPDX-FileCopyrightText: 2024-present Sunil Thaha <sthaha@redhat.com>
#
# SPDX-License-Identifier: MIT
import click
import logging
import time

from power_model.__about__ import __version__
from power_model import trainer

logger = logging.getLogger(__name__)


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
)
@click.version_option(version=__version__, prog_name="power-model")
@click.option(
    "--log-level",
    "-l",
    type=click.Choice(["debug", "info", "warn", "error"]),
    default="info",
    required=False,
)
def pm(log_level: str):
    level = getattr(logging, log_level.upper())
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(filename)s:%(lineno)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.debug("Log level set to %s", log_level)


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
        pipeline = trainer.load_pipeline(file)
        trainer.train(pipeline)
        click.echo("Training completed successfully.")

    except Exception as e:
        click.echo(f"An error occurred: {e}")


@pm.command()
@click.option(
    "-f",
    "--file",
    required=True,
    help="Path to the pipeline YAML file.",
    type=click.Path(exists=True),
)
def run(file):
    """Run models based on the provided pipeline configuration and compare the prediction against learning."""

    # try:
    pipeline = trainer.load_pipeline(file)
    predictor = trainer.Predictor(pipeline)

    try:
        while True:
            predictor.predict()
            time.sleep(1)
    except KeyboardInterrupt:
        click.echo("Exiting...")

    # except Exception as e:
    #    click.echo(f"An error occurred: {e}")


if __name__ == "__main__":
    pm()
