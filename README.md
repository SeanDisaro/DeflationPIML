# DeflationPIML
This repository is to test PDE deflation methods with neural networks.

## Installation
Make sure, that cuda is available!
Now make sure, that you are in the directory of this repository with your console. Then run

    pip install -r ./requirements.txt

to install the necessary packages. Afterwards run

    pip install -e .

to do some basic set up.
## Cite the paper

## Team

| Name        | Email                |
|-------------|----------------------|
| Sean Disarò | seandisaro@gmail.com |
| ...         | ...                  |

## License
This project is licensed under the GNU license. You may use it however you want!



## Some Comments Regarding Progress:
Experimented extentially with the src.DefDifONet.laplacian_2d.py architecture, but does not seem to work well. I'll try now a version where the derivative for the PDE is computed classically via auto diff.