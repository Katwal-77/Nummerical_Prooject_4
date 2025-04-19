# Discovering the Law Behind Real-World Data Using the Least Squares Method

**Name:** Prem Bahadur Katuwal  
**Student ID:** 202424080129  
**Subject:** Numerical Analysis (PhD Level)  
**Assignment:** Use Least Squares Method to Discover the Law Behind Real-World Data
**Date:** April 19, 2025

---

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset Description](#dataset-description)
3. [Visualizing the Data](#visualizing-the-data)
4. [Applying the Least Squares Method](#applying-the-least-squares-method)
5. [Results and Interpretation](#results-and-interpretation)
6. [Residual Analysis](#residual-analysis)
7. [Conclusion](#conclusion)
8. [References](#references)

---

## Introduction

The least squares method is a fundamental technique in numerical analysis for discovering the underlying law or relationship between variables in real-world data. This report demonstrates the application of the least squares method to a real-world-inspired dataset, detailing each step and providing visualizations for better understanding.

## Dataset Description

We consider a dataset relating **house size (in square feet)** to **house price (in $1000s)**, a classic example in regression analysis. The data points are as follows:

| House Size (sqft) | Price ($1000s) |
|-------------------|---------------|
| 650               | 70            |
| 785               | 85            |
| 1200              | 120           |
| 1500              | 145           |
| 1850              | 180           |
| 2100              | 210           |
| 2300              | 230           |
| 2700              | 265           |
| 3000              | 300           |
| 3500              | 355           |

## Visualizing the Data

Below is a scatter plot of the data, illustrating the positive correlation between house size and price:

![Scatter Plot](scatter_plot.png)

## Applying the Least Squares Method

The least squares method fits a model (here, a straight line) to the data by minimizing the sum of the squares of the vertical distances (residuals) between the observed values and the values predicted by the model.

The linear model is:

$$\text{Price} = m \times \text{Size} + c$$

where $m$ is the slope and $c$ is the intercept.

The best-fit parameters obtained are:
- **Slope ($m$):** 0.0991
- **Intercept ($c$):** 1.9382

Thus, the discovered law is:

> **Price = 0.10 × Size + 1.94 ($1000s)**

The fitted line is shown below:

![Fitted Line](fitted_line.png)

## Results and Interpretation

- The positive slope indicates that larger houses tend to have higher prices.
- The intercept represents the estimated price when the house size is zero (not meaningful in this context, but mathematically necessary).
- The fitted line closely follows the trend of the data, indicating a strong linear relationship.

## Residual Analysis

Residuals are the differences between the observed and predicted values. Analyzing residuals helps assess the goodness of fit.

| House Size (sqft) | Actual Price | Predicted Price | Residual |
|-------------------|--------------|-----------------|----------|
| 650               | 70           | 66.34           | 3.66     |
| 785               | 85           | 79.72           | 5.28     |
| 1200              | 120          | 120.84          | -0.84    |
| 1500              | 145          | 150.57          | -5.57    |
| 1850              | 180          | 185.25          | -5.25    |
| 2100              | 210          | 210.02          | -0.02    |
| 2300              | 230          | 229.84          | 0.16     |
| 2700              | 265          | 269.47          | -4.47    |
| 3000              | 300          | 299.20          | 0.80     |
| 3500              | 355          | 348.74          | 6.26     |

The plot below shows the residuals for each data point:

![Residuals](residuals.png)

Most residuals are small, indicating a good fit. The residuals are randomly scattered, suggesting the linear model is appropriate.

## Conclusion

- The least squares method successfully uncovered the linear relationship between house size and price.
- The discovered law can be used for prediction and analysis in real-world scenarios.
- Visualizations and residual analysis confirm the suitability of the model.

## References
- Numerical Analysis Textbooks
- [Wikipedia: Least Squares](https://en.wikipedia.org/wiki/Least_squares)
- Real-world housing data sources (for inspiration)

---

*This report was generated programmatically for educational purposes. All diagrams were created using Python (matplotlib).*
