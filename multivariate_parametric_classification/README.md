# Multivariate Parametric Classification
## Generating Random Samples
To generate random samples from given multivariate Gaussian Distrubitions ```numpy.random.multivariate_normal``` method is used.
## Parameter Estimation
To generate parameters from training examples, different ```Gaussian``` classes are used for different classes. And parameters of each class is estimated using corresponding training examples. Note that training example vectors are row-first.
### Mean Vector  
Numpy's ```numpy.mean``` method is used along ```axis=0``` to estimate the mean of the features for each class.
### Covariance Matrix
Since we are dealing with a small example, for loops are used for estimating the covariance matrix as a naive implementation using the estimated mean.
### Class Prior
Class priors are simply calculated by dividing number of examples belonging to the class and number of all examples.
## Classification
In the ```fit``` method of ```GaussianEstimator```, each example is grouped by its class, then for each class mean vectors, covariance matrices and class priors are estimated then using these parameters seperate ```Gaussian``` classes are initialized that holds the parameter information. Negative Log-Likelihood is used in these classes for classification of the examples. For one data point, NLL is used to get the likelihood using all the estimated Gaussians then ```numpy.argmax``` is used to predict the class. This is calculated in the ```get_discriminant``` function in ```Gaussian``` class. The book names this function discriminant function.  

To plot contours, for each feature of x, random values are sampled between the ```min - 0.25``` and ```max + 0.25``` of the features with a step size of 0.1. These samples are classified and results are plotted as a contour using matplotlib's ```matplotlib.pyplot.contourf``` method.  

## To run
```sh
$cd /to/path/0037975
$python3 train.py
```  
