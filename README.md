# PyPARC - A Python Package for Piecewise Affine Regression and Classification

# Contents

* [Package description](#description)

* [Installation](#install)

* [Basic usage](#basic-usage)

* [Contributors](#contributors)

* [Citing PARC](#bibliography)

* [License](#license)


<a name="description"></a>
## Package description 

**PyPARC** is a package for solving multivariate regression and classification problems using piecewise linear (affine) predictors over a polyhedral partition of the feature space. The underlying algorithm is called PARC (Piecewise Affine Regression and Classification) and is described in the following paper:

<a name="cite-Bem23"><a>
> [1] A. Bemporad, "[A piecewise linear regression and classification algorithm with application to learning and model predictive control of hybrid systems](http://cse.lab.imtlucca.it/~bemporad/publications/papers/ieeetac-parc.pdf)," *IEEE Transactions on Automatic Control*, vol. 68, pp. 3194–3209, June 2023. [[bib entry](#ref1)]

The algorithm alternates between:

1. Solving ridge regression problems (for numeric targets) and softmax regression problems (for categorical targets), and either softmax regression or cluster centroid computation for piecewise linear separation
2. Assigning the training points to different clusters on the basis of a criterion that balances prediction accuracy and piecewise-linear separability.

For earlier Python versions of the code, see <a href="http://cse.lab.imtlucca.it/~bemporad/parc/">here</a>.

<a name="install"></a>
## Installation

~~~python
pip install pyparc
~~~

<a name="basic-usage"></a>
## Basic usage

Say we want to fit a piecewise affine model on a dataset of 1000 samples $(x,y)$, where $x=(x_1,x_2)$ has two numeric components randomly generated between 0 and 1, and $y$ is obtained by the following nonlinear function of $x$ 
$$
	y(x_1,x_2)=\sin\left(4x_{1}-5\left(x_{2}-\frac{1}{2}\right)^2\right)+2x_2
$$

<img src="http://cse.lab.imtlucca.it/~bemporad/parc/fig4.png" alt="drawing" width=55%/>
<img src="http://cse.lab.imtlucca.it/~bemporad/parc/fig1.png" alt="drawing" width=40%/>


We use 80% of the data for training (`X_train`,`Y_train`) and 20% for testing the model (`X_test`,`Y_test`).

We want to train a piecewise affine model on a polyhedral partition with maximum 10 regions, with $\ell_2$-regularization coefficient $\alpha=10^{-4}$ and maximum 15 block-coordinate descent iterations of the algorithm:

~~~python
from parc.parc import PARC

predictor = PARC(K=10, alpha=1.0e-4, maxiter=15) # initialize PARC object

categorical = False # targets are numeric

# fit piecewise linear predictor
predictor.fit(X_train, Y_train) 

# make predictions on test data
Yhtest, _ = predictor.predict(X_test) 

# compute R2 scores
score_train = predictor.score(X_train, Y_train)  
score_test = predictor.score(X_test, Y_test)  
~~~

The resulting PWA model leads to the following partition
<img src="http://cse.lab.imtlucca.it/~bemporad/parc/fig3.png" alt="drawing" width=65%/>

and PWA function

<img src="http://cse.lab.imtlucca.it/~bemporad/parc/fig5.png" alt="drawing" width=50%/>
<img src="http://cse.lab.imtlucca.it/~bemporad/parc/fig2.png" alt="drawing" width=40%/>
               
<a name="contributors"><a>
## Contributors

This package was coded by Alberto Bemporad.


This software is distributed without any warranty. Please cite the above papers if you use this software.

<a name="bibliography"><a>
## Citing PARC

<a name="ref1"></a>

```
@article{Bem23,
    author={A. Bemporad},
    title={A Piecewise Linear Regression and Classification Algorithm with Application to Learning and Model Predictive Control of Hybrid Systems},
    journal={IEEE Transactions on Automatic Control},
    year=2023,
    month=jun,
    volume=68,
    number=6,
    pages={3194--3209},
}
```


<a name="license"><a>
## License

Apache 2.0

(C) 2021-2023 A. Bemporad
