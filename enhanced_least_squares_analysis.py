import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd

# Set up plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# =====================================================================
# Example 1: Linear Least Squares - House Size vs. Price
# =====================================================================

# Sample real-world-like data: House Size (sqft) vs. Price ($1000s)
house_size = np.array([650, 785, 1200, 1500, 1850, 2100, 2300, 2700, 3000, 3500])
house_price = np.array([70, 85, 120, 145, 180, 210, 230, 265, 300, 355])

def linear_least_squares_analysis():
    """Perform linear least squares analysis on house size vs. price data."""
    print("\n=== Linear Least Squares: House Size vs. Price ===")

    # Scatter plot of the data
    plt.figure()
    plt.scatter(house_size, house_price, color='blue', s=80, alpha=0.7, label='Data Points')
    plt.title('House Size vs. Price')
    plt.xlabel('House Size (sqft)')
    plt.ylabel('Price ($1000s)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('scatter_plot.png', dpi=300)
    plt.close()

    # Least Squares Linear Fit using matrix approach
    def least_squares_fit(x, y):
        """Implement least squares using the normal equations."""
        # Design matrix
        X = np.vstack([x, np.ones(len(x))]).T
        # Normal equations: (X^T X) β = X^T y
        # Solution: β = (X^T X)^(-1) X^T y
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        return beta[0], beta[1]  # slope, intercept

    slope, intercept = least_squares_fit(house_size, house_price)

    # Fitted line
    x_range = np.linspace(min(house_size) - 100, max(house_size) + 100, 1000)
    fitted_line = slope * x_range + intercept

    # Calculate fitted values for original data points
    fitted_price = slope * house_size + intercept

    # Calculate residuals
    residuals = house_price - fitted_price

    # Calculate goodness-of-fit metrics
    r2 = r2_score(house_price, fitted_price)
    rmse = np.sqrt(mean_squared_error(house_price, fitted_price))
    mae = mean_absolute_error(house_price, fitted_price)

    # Plot with fitted line
    plt.figure()
    plt.scatter(house_size, house_price, color='blue', s=80, alpha=0.7, label='Data Points')
    plt.plot(x_range, fitted_line, color='red', linewidth=2,
             label=f'Fitted Line: y = {slope:.4f}x + {intercept:.4f}')
    plt.title('Least Squares Fit: House Size vs. Price')
    plt.xlabel('House Size (sqft)')
    plt.ylabel('Price ($1000s)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add R² annotation
    plt.annotate(f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}',
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    plt.tight_layout()
    plt.savefig('fitted_line.png', dpi=300)
    plt.close()

    # Residuals plot
    plt.figure()
    plt.stem(house_size, residuals, linefmt='r-', markerfmt='bo', basefmt='k-')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.title('Residuals of Least Squares Fit')
    plt.xlabel('House Size (sqft)')
    plt.ylabel('Residual ($1000s)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('residuals.png', dpi=300)
    plt.close()

    # Confidence intervals plot
    # For simplicity, we'll use a basic approach to calculate confidence intervals
    n = len(house_size)
    x_mean = np.mean(house_size)
    t_value = 2.306  # 95% confidence for n-2 = 8 degrees of freedom

    # Standard error of the regression
    se = np.sqrt(np.sum(residuals**2) / (n - 2))

    # Standard error of the prediction
    x_std = np.sqrt(np.sum((house_size - x_mean)**2))
    se_fit = se * np.sqrt(1/n + (x_range - x_mean)**2 / x_std**2)

    plt.figure()
    plt.scatter(house_size, house_price, color='blue', s=80, alpha=0.7, label='Data Points')
    plt.plot(x_range, fitted_line, color='red', linewidth=2, label='Fitted Line')
    plt.fill_between(x_range, fitted_line - t_value * se_fit,
                     fitted_line + t_value * se_fit,
                     color='gray', alpha=0.2, label='95% Confidence Interval')
    plt.title('Least Squares Fit with Confidence Intervals')
    plt.xlabel('House Size (sqft)')
    plt.ylabel('Price ($1000s)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('confidence_intervals.png', dpi=300)
    plt.close()

    # Print results for report
    print('Least Squares Linear Fit Results:')
    print(f'Slope: {slope:.4f}')
    print(f'Intercept: {intercept:.4f}')
    print(f'R²: {r2:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'MAE: {mae:.4f}')
    print('\nInterpreted Law:')
    print(f'Price = {slope:.4f} * Size + {intercept:.4f} ($1000s)')

    # Create a table of actual vs. predicted values
    print('\nActual vs. Predicted Values:')
    table = pd.DataFrame({
        'House Size (sqft)': house_size,
        'Actual Price ($1000s)': house_price,
        'Predicted Price ($1000s)': fitted_price.round(2),
        'Residual ($1000s)': residuals.round(2)
    })
    print(table)

    return slope, intercept, r2, rmse, mae

# =====================================================================
# Example 2: Polynomial Least Squares - Temperature Data
# =====================================================================

# Monthly temperature data (northern hemisphere)
months = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
temperatures = np.array([-5.2, -3.8, 2.4, 8.7, 14.5, 19.2, 21.8, 20.9, 16.3, 10.1, 4.2, -2.5])

def polynomial_least_squares_analysis():
    """Perform polynomial least squares analysis on temperature data."""
    print("\n=== Polynomial Least Squares: Monthly Temperatures ===")

    # Scatter plot of the data
    plt.figure()
    plt.scatter(months, temperatures, color='green', s=80, alpha=0.7, label='Data Points')
    plt.title('Monthly Average Temperatures')
    plt.xlabel('Month')
    plt.ylabel('Temperature (°C)')
    plt.xticks(months)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('temperature_data.png', dpi=300)
    plt.close()

    # Fit polynomials of different degrees
    max_degree = 6
    x_smooth = np.linspace(0.5, 12.5, 1000)
    r2_values = []
    adj_r2_values = []

    plt.figure(figsize=(12, 8))
    plt.scatter(months, temperatures, color='green', s=80, alpha=0.7, label='Data Points')

    for degree in range(1, max_degree + 1):
        # Fit polynomial
        coeffs = np.polyfit(months, temperatures, degree)
        poly = np.poly1d(coeffs)

        # Calculate fitted values
        y_fit = poly(months)
        y_smooth = poly(x_smooth)

        # Calculate R² and adjusted R²
        r2 = r2_score(temperatures, y_fit)
        n = len(months)
        p = degree + 1  # number of parameters
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        r2_values.append(r2)
        adj_r2_values.append(adj_r2)

        # Plot the polynomial fit
        if degree in [1, 2, 4]:  # Plot only selected degrees for clarity
            plt.plot(x_smooth, y_smooth, linewidth=2,
                     label=f'Degree {degree}: R² = {r2:.3f}')

    plt.title('Polynomial Fits of Different Degrees')
    plt.xlabel('Month')
    plt.ylabel('Temperature (°C)')
    plt.xticks(months)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('temperature_poly_comparison.png', dpi=300)
    plt.close()

    # Plot the best fit (degree 4)
    best_degree = 4
    coeffs = np.polyfit(months, temperatures, best_degree)
    poly = np.poly1d(coeffs)
    y_smooth = poly(x_smooth)

    plt.figure()
    plt.scatter(months, temperatures, color='green', s=80, alpha=0.7, label='Data Points')
    plt.plot(x_smooth, y_smooth, color='purple', linewidth=2,
             label=f'Degree {best_degree} Polynomial')
    plt.title(f'Best Polynomial Fit (Degree {best_degree})')
    plt.xlabel('Month')
    plt.ylabel('Temperature (°C)')
    plt.xticks(months)
    plt.grid(True, alpha=0.3)

    # Add equation annotation
    eq_text = f"T = {coeffs[0]:.3f}x⁴"
    for i, c in enumerate(coeffs[1:-1], 1):
        power = best_degree - i
        sign = "+" if c >= 0 else ""
        eq_text += f" {sign} {c:.3f}x{'^' + str(power) if power > 1 else ''}"
    sign = "+" if coeffs[-1] >= 0 else ""
    eq_text += f" {sign} {coeffs[-1]:.3f}"

    plt.annotate(eq_text, xy=(0.5, 0.05), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    plt.legend()
    plt.tight_layout()
    plt.savefig('temperature_fit.png', dpi=300)
    plt.close()

    # Plot R² and adjusted R² vs. polynomial degree
    plt.figure()
    plt.plot(range(1, max_degree + 1), r2_values, 'bo-', linewidth=2, label='R²')
    plt.plot(range(1, max_degree + 1), adj_r2_values, 'ro-', linewidth=2, label='Adjusted R²')
    plt.title('Model Selection: R² vs. Polynomial Degree')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Coefficient of Determination')
    plt.xticks(range(1, max_degree + 1))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('temperature_r2.png', dpi=300)
    plt.close()

    # Print results
    print('Polynomial Fit Results:')
    print(f'Best degree: {best_degree}')
    print(f'Coefficients: {coeffs}')
    print('\nR² values for different degrees:')
    for degree, r2, adj_r2 in zip(range(1, max_degree + 1), r2_values, adj_r2_values):
        print(f'Degree {degree}: R² = {r2:.4f}, Adjusted R² = {adj_r2:.4f}')

    print('\nInterpreted Law (Degree 4):')
    print(eq_text)

    return coeffs, r2_values, adj_r2_values

# =====================================================================
# Example 3: Non-Linear Least Squares - Population Growth
# =====================================================================

# Population growth data
years = np.array([1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020])
population = np.array([1.65, 1.92, 2.22, 2.59, 2.85, 3.17, 3.65, 4.07, 4.45, 5.31, 6.12, 6.93, 7.79])

def exponential_growth(x, a, b):
    """Exponential growth model: a * exp(b * x)"""
    return a * np.exp(b * x)

def non_linear_least_squares_analysis():
    """Perform non-linear least squares analysis on population growth data."""
    print("\n=== Non-Linear Least Squares: Population Growth ===")

    # Normalize years to start from 0
    years_normalized = years - years[0]

    # Scatter plot of the data
    plt.figure()
    plt.scatter(years, population, color='orange', s=80, alpha=0.7, label='Data Points')
    plt.title('World Population Growth')
    plt.xlabel('Year')
    plt.ylabel('Population (billions)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('population_data.png', dpi=300)
    plt.close()

    # Method 1: Linearization through logarithm
    log_population = np.log(population)

    # Linear fit on log-transformed data
    slope, intercept = np.polyfit(years_normalized, log_population, 1)

    # Convert back to exponential form
    a_linear = np.exp(intercept)
    b_linear = slope

    # Method 2: Direct non-linear fit using curve_fit
    params, covariance = curve_fit(exponential_growth, years_normalized, population, p0=[1.65, 0.01])
    a_nonlinear, b_nonlinear = params

    # Generate smooth curves for plotting
    years_smooth = np.linspace(min(years_normalized) - 10, max(years_normalized) + 10, 1000)
    years_actual = years_smooth + years[0]

    # Fitted curves
    pop_linear = a_linear * np.exp(b_linear * years_smooth)
    pop_nonlinear = exponential_growth(years_smooth, a_nonlinear, b_nonlinear)

    # Calculate fitted values for original data points
    fitted_linear = a_linear * np.exp(b_linear * years_normalized)
    fitted_nonlinear = exponential_growth(years_normalized, a_nonlinear, b_nonlinear)

    # Calculate R² for both methods
    r2_linear = r2_score(population, fitted_linear)
    r2_nonlinear = r2_score(population, fitted_nonlinear)

    # Plot both fits
    plt.figure()
    plt.scatter(years, population, color='orange', s=80, alpha=0.7, label='Data Points')
    plt.plot(years_actual, pop_linear, 'b-', linewidth=2,
             label=f'Linearized Fit: P = {a_linear:.2f}e^({b_linear:.4f}t), R² = {r2_linear:.4f}')
    plt.plot(years_actual, pop_nonlinear, 'r--', linewidth=2,
             label=f'Non-Linear Fit: P = {a_nonlinear:.2f}e^({b_nonlinear:.4f}t), R² = {r2_nonlinear:.4f}')
    plt.title('Exponential Growth Models for Population')
    plt.xlabel('Year')
    plt.ylabel('Population (billions)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('population_growth.png', dpi=300)
    plt.close()

    # Extrapolation to future years
    future_years = np.array([2030, 2040, 2050, 2060, 2070])
    future_normalized = future_years - years[0]

    future_pop_linear = a_linear * np.exp(b_linear * future_normalized)
    future_pop_nonlinear = exponential_growth(future_normalized, a_nonlinear, b_nonlinear)

    # Print results
    print('Exponential Growth Model Results:')
    print('\nMethod 1: Linearization through logarithm')
    print(f'Initial population (P₀): {a_linear:.4f} billion')
    print(f'Growth rate (r): {b_linear:.4f} per year')
    print(f'Doubling time: {np.log(2)/b_linear:.1f} years')
    print(f'R²: {r2_linear:.4f}')
    print(f'Equation: P(t) = {a_linear:.4f} * e^({b_linear:.4f}t), where t = years since {years[0]}')

    print('\nMethod 2: Direct non-linear fit')
    print(f'Initial population (P₀): {a_nonlinear:.4f} billion')
    print(f'Growth rate (r): {b_nonlinear:.4f} per year')
    print(f'Doubling time: {np.log(2)/b_nonlinear:.1f} years')
    print(f'R²: {r2_nonlinear:.4f}')
    print(f'Equation: P(t) = {a_nonlinear:.4f} * e^({b_nonlinear:.4f}t), where t = years since {years[0]}')

    print('\nPopulation Projections:')
    projection_table = pd.DataFrame({
        'Year': future_years,
        'Linearized Model (billions)': future_pop_linear.round(2),
        'Non-Linear Model (billions)': future_pop_nonlinear.round(2)
    })
    print(projection_table)

    return (a_linear, b_linear, r2_linear), (a_nonlinear, b_nonlinear, r2_nonlinear)

# =====================================================================
# Main execution
# =====================================================================

if __name__ == "__main__":
    print("=== Least Squares Method Analysis ===")
    print("Analyzing real-world data to discover underlying laws...")

    # Example 1: Linear Least Squares
    linear_results = linear_least_squares_analysis()

    # Example 2: Polynomial Least Squares
    polynomial_results = polynomial_least_squares_analysis()

    # Example 3: Non-Linear Least Squares
    nonlinear_results = non_linear_least_squares_analysis()

    print("\n=== Analysis Complete ===")
    print("All results and visualizations have been saved.")
