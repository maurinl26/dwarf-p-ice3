# -*- coding: utf-8 -*-
import numpy as np
import logging
import sys
import os
import xarray as xr
import typer

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logging.getLogger()

################## APP ####################################
app = typer.Typer()


def extract_testsprogs_data(
    inputdir: str, output_file: str, nproma: int = 23, ngpblks: int = 296
):
    NotImplemented


if __name__ == "__main__":
    app()
