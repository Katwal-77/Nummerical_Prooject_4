import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Sample real-world-like data: House Size (sqft) vs. Price ($1000s)
# (This is synthetic but mimics real-world behavior)
house_size = np.array([650, 785, 1200, 1500, 1850, 2100, 2300, 2700, 3000, 3500])
house_price = np.array([70, 85, 120, 145, 180, 210, 230, 265, 300, 355])

# Scatter plot of the data
plt.figure(figsize=(8, 6))
plt.scatter(house_size, house_price, color='blue', label='Data Points')
plt.title('House Size vs. Price')
plt.xlabel('House Size (sqft)')
plt.ylabel('Price ($1000s)')
plt.grid(True)
plt.legend()
plt.savefig('scatter_plot.png')
plt.close()

# Least Squares Linear Fit
def least_squares_fit(x, y):
    n = len(x)
    A = np.vstack([x, np.ones(n)]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c

slope, intercept = least_squares_fit(house_size, house_price)

# Fitted line
fitted_price = slope * house_size + intercept

# Plot with fitted line
plt.figure(figsize=(8, 6))
plt.scatter(house_size, house_price, color='blue', label='Data Points')
plt.plot(house_size, fitted_price, color='red', linewidth=2, label=f'Fitted Line: y = {slope:.2f}x + {intercept:.2f}')
plt.title('Least Squares Fit: House Size vs. Price')
plt.xlabel('House Size (sqft)')
plt.ylabel('Price ($1000s)')
plt.grid(True)
plt.legend()
plt.savefig('fitted_line.png')
plt.close()

# Residuals plot
residuals = house_price - fitted_price
plt.figure(figsize=(8, 4))
plt.stem(house_size, residuals)
plt.title('Residuals of Least Squares Fit')
plt.xlabel('House Size (sqft)')
plt.ylabel('Residual ($1000s)')
plt.grid(True)
plt.savefig('residuals.png')
plt.close()

# Print results for report
def print_results():
    print('Least Squares Linear Fit Results:')
    print(f'Slope: {slope:.4f}')
    print(f'Intercept: {intercept:.4f}')
    print('\nInterpreted Law:')
    print(f'Price = {slope:.2f} * Size + {intercept:.2f} ($1000s)')
    print('\nResiduals (errors for each data point):')
    print(residuals)

if __name__ == "__main__":
    print_results()
