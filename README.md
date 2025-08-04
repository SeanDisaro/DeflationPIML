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
The experiment described in the paper runs, if you run `python main.py` with the environment described above.

_____________________________________________________
Contains solutions for the LDG problem from the paper which were generated with FEM. You can just try them out in a notebook. The coordinates needed for the points are saved in `data_test.mat`

```
📦data
 ┗ 📂Reduced2DimLDG
 ┃ ┗ 📂trueSolution
 ┃ ┃ ┣ 📜data_15112024.mat
 ┃ ┃ ┣ 📜data_D2.mat
 ┃ ┃ ┣ 📜data_R1.mat
 ┃ ┃ ┣ 📜data_R2.mat
 ┃ ┃ ┣ 📜data_R3.mat
 ┃ ┃ ┣ 📜data_R4.mat
 ┃ ┃ ┗ 📜data_test.mat
 ```
________________________________________________________


`src` contains model implementations and loss functions and other usefull functions. `architectures` contains neural network architectures which were used for different experiments. The architecture from the paper is implemented in `twoDimFixedFeatureONet`. `lossFunctions` contains code for PIML and Deflation loss functions.

```
📦src
 ┣ 📂architectures
 ┃ ┣ 📂DefDifONets
 ┃ ┃ ┣ 📜laplacian_2d.py
 ┃ ┃ ┣ 📜twoDimFeatureONet.py
 ┃ ┃ ┗ 📜twoDimFixedFeatureONet.py
 ┣ 📂lossFunctions
 ┃ ┣ 📜derivativeLoss.py
 ┃ ┣ 📜Reduced2DimLDG.py
 ┣ 📜differentialOperators.py
 ```

________________________________________________________


`starDomainExtrapolation` contains tools for the star domain function extrapolation described in the paper for implementing the mentioned Dirichlet hard constraint.

```
📦starDomainExtrapolation
 ┗📜starDomain.py
```

________________________________________________________

`tests` contains various experiments to test out different architectures. Each test gets its own directory and has a `run.py`, `testing.py` and a `training.py`. `run.py` provides a function, which gets called by the `main.py` file to run the respective experiment. `testing.py` implements various testing functions which get called in `run.py`. This includes for the most part some plotting functions. `training.py` defines the training loop of the experiment. The experiment from the paper can be found in `Reduced2DimLDG_FixedBranchFeatures`. The other experiments did not work well at some point and were abandoned.

```
📦tests
┣ 📂pictures
┃ ┣ 📂Reduced2DimLDG_ApproxDerivativeWithNN
┃ ┣ 📂Reduced2DimLDG_AutoGradDerivative
┃ ┗ 📂Reduced2DimLDG_FixedBranchFeatures
┣ 📂Reduced2DimLDG_ApproxDerivativeWithNN
┃ ┣ 📜run.py
┃ ┣ 📜testing.py
┃ ┣ 📜training.py
┣ 📂Reduced2DimLDG_AutoGradDerivative
┃ ┣ 📜run.py
┃ ┣ 📜testing.py
┃ ┗ 📜training.py
┣ 📂Reduced2DimLDG_FixedBranchFeatures
┃ ┣ 📜run.py
┃ ┣ 📜testing.py
┃ ┗ 📜training.py
```




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