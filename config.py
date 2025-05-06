import os
from pathlib import Path, PurePath


repositoryPath: PurePath = os.getcwd()

pathTrainedModels: PurePath = PurePath(repositoryPath, "models")
loggingFile: PurePath = PurePath(repositoryPath, "logging.log")
testsFolder: PurePath = PurePath(repositoryPath, "tests")
plotFolder: PurePath = PurePath(testsFolder, "pictures")
