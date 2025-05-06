import numpy as np
import torch
import deepxde as dde
from src.architectures.DefDifONets.laplacian_2d import Laplacian_2D_DefDifONet
from logging_config import setup_logging
import logging
from tests.Reduced2DimLDG.run import run

logger = setup_logging()

#logger = logging.getLogger(__name__)


if __name__ == "__main__":
    run()