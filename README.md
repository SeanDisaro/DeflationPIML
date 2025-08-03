# DeflationPIML
This repository is to test PDE deflation methods with neural networks.

## Installation
Use python 3.12.
Make sure, that cuda is available!
Now make sure, that you are in the directory of this repository with your console. Then run

    pip install -r ./requirements.txt

to install the necessary packages. Afterwards run

    pip install -e .

to do some basic set up.

## Structure of the repo



## Cite the paper
[Not Uploaded Yet]

## Link to Repo where Finite Element Solutions were generated
[Insert Link]

## Team

| Name        | Email                 |
|-------------|-----------------------|
| Sean Disarò | seandisaro@gmail.com  |
| Aras Bacho  | bacho@caltech.edu     |
| Ruma Maity  | rumamaity081@gmail.com|

## License
This project is licensed under the GNU license. You may use it however you want!



## Some Comments Regarding Progress:
Experimented extentially with the src.DefDifONet.laplacian_2d.py architecture, but does not seem to work well. I'll try now a version where the derivative for the PDE is computed classically via auto diff.