# Project 2

## Model Selection

Implement generic k-fold cross-validation and bootstrapping model selection methods.

In your README, answer the following questions:

** Do your cross-validation and bootstrapping model selectors agree with a simpler model selector like AIC in simple cases (like linear regression)?**
  - Yes. $R^2$ of Ridge Regression is 0.92 and mean $R^2$ of our cross-validation is 0.79 and bootstrapping model is 0.81.
 
** In what cases might the methods you've written fail or give incorrect or undesirable results?**
  1. High-Dimensional Data or Multicollinearity:
    -	Cross-validation may overestimate performance if features are highly correlated.
    -	Bootstrapping might not capture variability well with duplicate features.
	2. Imbalanced Data:
	  -	Both methods can produce misleading results when the target variable is heavily skewed.
	3. Small Dataset:
	  -	Cross-validation might struggle to split data effectively, leading to unstable results.
	  -	Bootstrapping may produce overly optimistic results by repeatedly sampling the same data points.
	4. Outliers:
	  -	Outliers can distort performance metrics and affect evaluation accuracy.
	5. Complex Models:
	  -	Non-linear interactions or complex dependencies might not be fully captured by either method.

** What could you implement given more time to mitigate these cases or help users of your methods?**
  -

** What parameters have you exposed to your users in order to use your model selectors.**
  1. Cross-Validation:
	  -	cv_folds: Number of folds.
	  -	scoring_metric: Metric for evaluation.
	  -	shuffle: Whether to shuffle data before splitting.
	2. Bootstrapping:
	  -	n_iterations: Number of bootstrap iterations.
	  -	random_seed: Seed for reproducibility.
	  -	metric: Metric to evaluate performance.
	3. Hyperparameter Tuning:
	  -	alpha_range: Range of regularization parameters for Ridge/Lasso.
	4. Feature Selection:
	  -	correlation_threshold: Minimum correlation value for feature inclusion.
	5. Model Selection:
	  -	Options for Ridge, Lasso, or Linear Regression.

  By exposing these parameters, users can customize the evaluation process to fit their specific data and model requirements, ensuring better control over regularization and performance evaluation.

See sections 7.10-7.11 of Elements of Statistical Learning and the lecture notes. Pay particular attention to Section 7.10.2.

As usual, above-and-beyond efforts will be considered for bonus points.
