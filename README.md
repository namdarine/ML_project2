# Project 2

## Model Selection

Implement generic k-fold cross-validation and bootstrapping model selection methods.

How to use: This code uses packages such as pandas, numpy, statemodels, seaborn, and matplotlib. You need to install each packages. Open CMD -> pip install numpy -> pip install pandas -> pip install statsmodels -> pip install seaborn -> pip install matplotlib. After installing the package you can run each code blocks from top to bottom.

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

  1. Improve Data Quality:
     - Use better imputation methods and add more relevant data, like market trends or historical prices, to provide richer context.
  2. Try Better Models:
     - Experiment with advanced models like gradient boosting, neural networks, or model ensembles to capture complex relationships.
  3. Improve Evaluation:
     - Use smarter validation methods, like stratified k-fold, and analyze residuals to catch biases or patterns we missed.
  4. Simplify Use:
     - Build an interactive dashboard for predictions and allow users to test "what-if" scenarios for better decision-making.
  5. Fix Limitations:
     - Address multicollinearity with Elastic Net regularization and confirm model stability with thorough bootstrapping.
  6. Automate & Scale:
      - Automate preprocessing and deploy the model as an API for real-time predictions.

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
