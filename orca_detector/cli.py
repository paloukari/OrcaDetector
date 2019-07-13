# -*- coding: utf-8 -*-

"""Console script for orca_detector."""
import click
import os
import sys

from live_feed_listener import live_feed_inference
from database_parser import read_files_and_extract_features
#from training import train
#from inference import infer
#from vggish_model import test_VGGish_model
#from logreg_model import test_logistic_regression_model

os.environ['LC_ALL'] = 'C.UTF-8'
os.environ['LANG'] = 'C.UTF-8'

@click.group()
def main(args=None):
    click.echo('OrcaDetector - W251 (Summer 2019)')
    pass


if __name__ == "__main__":

    main.add_command(read_files_and_extract_features, name="features")
    #main.add_command(train)
    #main.add_command(infer)
    main.add_command(live_feed_inference, name="infer-live")
    #main.add_command(test_VGGish_model, name="test-VGGish")
    #main.add_command(test_logistic_regression_model, name="test-LR")
    sys.exit(main())  # pragma: no cover
