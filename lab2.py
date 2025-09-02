import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import statistics as stats
from pathlib import Path

# Set up logging to display information messages
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def check_file_exists(file_path):
    if not Path(file_path).is_file():
        logging.error(f"Excel file not found at: {file_path}")
        raise FileNotFoundError(f"Excel file not found at: {file_path}")
    logging.info(f"Excel file found: {file_path}")

# A1: Matrix Operations on Purchase Data
def analyze_purchase_matrix(file_path):
    # Load Purchase Data from Excel
    data = pd.read_excel(file_path, sheet_name="Purchase data")
    
    # Extract features (Candies, Mangoes, Milk Packets) and target (Payment)
    X = data.iloc[:, 1:4].values  # Feature matrix
    y = data.iloc[:, 4].values    # Payment vector
    
    # Calculate matrix properties
    num_dimensions = X.shape[1]
    num_vectors = X.shape[0]
    matrix_rank = np.linalg.matrix_rank(X)
    
    # Compute pseudo-inverse and product costs
    X_pinv = np.linalg.pinv(X)
    product_costs = np.dot(X_pinv, y)
    
    # Log results
    logging.info(f"Purchase Data Analysis:")
    logging.info(f"Dimensions of feature matrix: {num_dimensions}")
    logging.info(f"Number of data points: {num_vectors}")
    logging.info(f"Matrix rank: {matrix_rank}")
    logging.info(f"Pseudo-inverse:\n{X_pinv}")
    logging.info(f"Cost per product:\n{product_costs}")
    
    return X, y, X_pinv, product_costs

#  A2: Customer Classification
def classify_customers(file_path):
    # Load Purchase Data
    data = pd.read_excel(file_path, sheet_name="Purchase data")
    
    # Label customers based on payment amount
    data["Category"] = data["Payment (Rs)"].apply(lambda x: "Rich" if x > 200 else "Poor")
    
    # Select relevant columns for display
    result = data[["Customer", "Candies (#)", "Mangoes (Kg)", "Milk Packets (#)", "Payment (Rs)", "Category"]]
    
    # Log classified data
    logging.info(f"Customer Classifications:\n{result}")
    
    return result

# A3: Stock Price Analysis
def analyze_stock_prices(file_path):
    # Load IRCTC Stock Price data
    stock_data = pd.read_excel(file_path, sheet_name="IRCTC Stock Price")
    prices = stock_data["Price"].values
    
    # Calculate overall mean and variance
    mean_price = stats.mean(prices)
    variance_price = stats.variance(prices)
    
    # Analyze Wednesday prices
    wed_prices = stock_data[stock_data["Day"] == "Wed"]["Price"].astype(float)
    wed_mean = stats.mean(wed_prices) if len(wed_prices) > 0 else 0
    
    # Analyze April prices
    apr_prices = stock_data[stock_data["Month"] == "Apr"]["Price"].astype(float)
    apr_mean = stats.mean(apr_prices) if len(apr_prices) > 0 else 0
    
    # Analyze percentage change on Wednesdays
    wed_changes = pd.to_numeric(stock_data[stock_data["Day"] == "Wed"]["Chg%"], errors="coerce")
    wed_change_mean = stats.mean(wed_changes.dropna()) if len(wed_changes.dropna()) > 0 else 0
    
    # Calculate probability of profit on Wednesdays
    profit_prob = (wed_changes > 0).mean() if len(wed_changes.dropna()) > 0 else 0
    
    # Log results
    logging.info(f"Stock Price Analysis:")
    logging.info(f"Overall Mean Price: {mean_price:.2f}")
    logging.info(f"Price Variance: {variance_price:.2f}")
    logging.info(f"Wednesday Mean Price: {wed_mean:.2f}")
    logging.info(f"April Mean Price: {apr_mean:.2f}")
    logging.info(f"Wednesday Mean Change %: {wed_change_mean:.2f}")
    logging.info(f"Probability of Profit on Wednesday: {profit_prob:.2f}")
    
    # Plot price change vs. day of the week
    stock_data["Day_Numeric"] = stock_data["Day"].map({"Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, "Fri": 5, "Sat": 6, "Sun": 7})
    changes = pd.to_numeric(stock_data["Chg%"], errors="coerce")
    plt.figure(figsize=(8, 6))
    plt.scatter(stock_data["Day_Numeric"], changes, alpha=0.6)
    plt.xlabel("Day of the Week (1=Mon, 7=Sun)")
    plt.ylabel("Price Change (%)")
    plt.title("Stock Price Change vs. Day of the Week")
    plt.grid(True)
    plt.show()
    
    return mean_price, variance_price, wed_mean, apr_mean, profit_prob

# A4: Thyroid Data Cleaning and Normalization
def process_thyroid_data(file_path):
    # Load Thyroid Data
    data = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")
    
    # Replace '?' with NaN
    data.replace("?", np.nan, inplace=True)
    
    # Numeric columns for processing
    numeric_cols = ["TSH", "T3", "TT4", "T4U", "FTI", "TBG"]
    data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors="coerce")
    
    # Detect outliers using IQR
    def find_outliers(col_data):
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (col_data < lower_bound) | (col_data > upper_bound)
    
    outlier_cols = [col for col in numeric_cols if find_outliers(data[col]).sum() > 0]
    logging.info(f"Columns with outliers: {outlier_cols}")
    
    # Handle missing values
    for col in numeric_cols:
        if col in outlier_cols:
            data[col].fillna(data[col].median(), inplace=True)
            logging.info(f"Filled missing values in {col} with median")
        else:
            data[col].fillna(data[col].mean(), inplace=True)
            logging.info(f"Filled missing values in {col} with mean")
    
    # Handle missing values in categorical columns
    categorical_cols = data.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        data[col].fillna(data[col].mode()[0], inplace=True)
        logging.info(f"Filled missing values in {col} with mode")
    
    # Normalize numeric columns (Min-Max scaling)
    for col in numeric_cols:
        col_min = data[col].min()
        col_max = data[col].max()
        if col_min != col_max:
            data[col] = (data[col] - col_min) / (col_max - col_min)
            logging.info(f"Normalized column {col}")
        else:
            logging.warning(f"Skipped normalization for {col} (constant values)")
    
    # Save cleaned data
    output_path = "Imputed_data.xlsx"
    data.to_excel(output_path, index=False, engine="openpyxl")
    logging.info(f"Cleaned data saved to {output_path}")
    
    return data, numeric_cols

# A5: Similarity Computations
def compute_similarities(data, binary_cols, numeric_cols):
    # Convert binary columns to 1/0
    data[binary_cols] = data[binary_cols].replace({"t": 1, "f": 0})
    
    # Compute SMC and Jaccard for binary data
    def compute_smc_jc(v1, v2):
        m00 = sum((v1 == 0) & (v2 == 0))
        m11 = sum((v1 == 1) & (v2 == 1))
        m01 = sum((v1 == 0) & (v2 == 1))
        m10 = sum((v1 == 1) & (v2 == 0))
        smc = (m11 + m00) / (m11 + m00 + m10 + m01) if (m11 + m00 + m10 + m01) > 0 else 0
        jc = m11 / (m11 + m10 + m01) if (m11 + m10 + m01) > 0 else 0
        return smc, jc
    
    # Compute Cosine similarity for numeric data
    def compute_cosine(v1, v2):
        dot = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        return dot / (norm_v1 * norm_v2) if norm_v1 > 0 and norm_v2 > 0 else 0
    
    # Compute similarity matrices for first 20 rows
    subset = data.loc[:19, binary_cols].values
    num_subset = data.loc[:19, numeric_cols].values
    n = len(subset)
    smc_matrix = np.zeros((n, n))
    jc_matrix = np.zeros((n, n))
    cos_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                smc_matrix[i, j] = jc_matrix[i, j] = cos_matrix[i, j] = 1
            else:
                smc_matrix[i, j], jc_matrix[i, j] = compute_smc_jc(subset[i], subset[j])
                cos_matrix[i, j] = compute_cosine(num_subset[i], num_subset[j])
    
    # Plot heatmaps
    def plot_heatmap(matrix, title):
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title(title)
        plt.show()
    
    logging.info("Plotting similarity heatmaps...")
    plot_heatmap(smc_matrix, "Simple Matching Coefficient Heatmap")
    plot_heatmap(jc_matrix, "Jaccard Coefficient Heatmap")
    plot_heatmap(cos_matrix, "Cosine Similarity Heatmap")
    
    # Log sample similarities
    smc, jc = compute_smc_jc(subset[0], subset[1])
    cos = compute_cosine(num_subset[0], num_subset[1])
    logging.info(f"Sample SMC (rows 0 vs 1): {smc:.2f}")
    logging.info(f"Sample Jaccard (rows 0 vs 1): {jc:.2f}")
    logging.info(f"Sample Cosine (rows 0 vs 1): {cos:.2f}")
    
    return smc_matrix, jc_matrix, cos_matrix

# Main execution
def main():
    file_path = "Lab Session Data (1).xlsx"
    check_file_exists(file_path)
    
    # Run all tasks
    logging.info("Starting Purchase Data Analysis...")
    X, y, X_pinv, product_costs = analyze_purchase_matrix(file_path)
    
    logging.info("\nStarting Customer Classification...")
    classified_data = classify_customers(file_path)
    
    logging.info("\nStarting Stock Price Analysis...")
    mean_price, variance_price, wed_mean, apr_mean, profit_prob = analyze_stock_prices(file_path)
    
    logging.info("\nStarting Thyroid Data Processing...")
    thyroid_data, numeric_cols = process_thyroid_data(file_path)
    
    binary_cols = [
        "on thyroxine", "query on thyroxine", "on antithyroid medication", "sick",
        "pregnant", "thyroid surgery", "I131 treatment", "query hypothyroid", 
        "query hyperthyroid", "lithium", "goitre", "tumor", "hypopituitary", "psych"
    ]
    
    logging.info("\nStarting Similarity Computations...")
    smc_matrix, jc_matrix, cos_matrix = compute_similarities(thyroid_data, binary_cols, numeric_cols)
    
    logging.info("All tasks completed successfully.")

if __name__ == "__main__":
    main()
