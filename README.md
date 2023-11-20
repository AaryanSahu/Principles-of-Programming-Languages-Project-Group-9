# Principles-of-Programming-Languages-Group-Project

# Bayesian House Price Prediction with Pyro

## Problem Statement

**Original Statement:** The goal is to predict house prices using a Bayesian linear regression model implemented in Pyro.

**POPL Angle:** The original problem involves probabilistic modeling and inference using Pyro, a probabilistic programming library. It's a Bayesian approach to regression, providing uncertainty estimates along with predictions. The problem is novel in its use of Pyro and Bayesian methods for house price prediction, differentiating it from traditional linear regression.

## Software Architecture

- The architecture involves data preprocessing using pandas, PyTorch for tensor operations, Pyro for probabilistic modeling, and matplotlib/seaborn for result visualization.
- No explicit client-server architecture; it's a standalone script.
- Testing is conducted locally, assessing model performance using R-squared and visualization.
- No database is involved; the dataset is fetched using scikit-learn.

## POPL Aspects

1. **Probabilistic Programming (Pyro):** The use of Pyro for probabilistic modeling introduces the concept of probabilistic programming, allowing the definition of Bayesian models with uncertainty.
2. **Plate Notation:** The `pyro.plate` construct is used for vectorized computation, improving efficiency in the probabilistic model.
3. **Bayesian Linear Regression:** The model includes priors for slope, intercept, and sigma, embodying Bayesian concepts in linear regression.
4. **Markov Chain Monte Carlo (MCMC):** NUTS (No-U-Turn Sampler) is utilized for MCMC inference, capturing the posterior distribution of parameters.
5. **Parameter Store Management:** The `pyro.clear_param_store()` ensures a clean slate for each run, managing the parameter store in a Bayesian context.
6. **Posterior Analysis:** Visualization of posterior distributions provides insights into uncertainty and parameter estimates.
7. **Comparison with Traditional Regression:** The script includes a section comparing Bayesian regression results with a traditional linear regression using scikit-learn.

## Results and Tests

- **Dataset:** California housing dataset is used, split into training and testing sets.
- **Benchmark:** R-squared is calculated to assess model performance. Visualizations include histograms of posterior distributions and scatter plots comparing predicted and true house prices.
- **Validation:** The comparison with traditional linear regression acts as a validation point, demonstrating the benefits of the Bayesian approach in capturing uncertainty.

## Potential for Future Work

- **Hyperparameter Tuning:** Explore sensitivity to priors and hyperparameters for better model performance.
- **Feature Engineering:** Experiment with additional features or transformations to improve predictive accuracy.
- **Ensemble Methods:** Investigate ensemble methods or model averaging to enhance robustness.
- **Online Learning:** Explore possibilities for online learning and continuous model improvement.
- **Integration with External Data:** Incorporate external data sources for richer feature sets.
- **Deployment:** Consider deployment strategies for the model, possibly as a web service or API.
- **Explanability:** Integrate tools or techniques for explaining model predictions to end-users.

This comprehensive approach covers both the technical implementation details and potential areas for further exploration, aligning with principles of Probabilistic Programming and Machine Learning.

## Group-Members
- **Aryan Sahu :** 2021A7PS2832G
- **Anuj Nethwewala:** 2021A7PS2716G
- **Imaad Momin:** 2021A7PS2066G
- **Subhradip Maity:** 2021A7PS2983G


  ## File Organization:
- [Dataset](PoPL/Dataset/california_housing_data.csv)
- [code-external](PoPL/code-external/Test_code.ipynb)
- [code-orig](PoPL/code-orig/BayesianRegression2.ipynb)
- [result](PoPL/result)

  ## How to Run
-Open the Python Notebook on any python environment.(**Ideally Google Colab**).
-Load the Dataset from [Dataset](PoPL/Dataset/california_housing_data.csv).
-Test the code and generate graphs.

