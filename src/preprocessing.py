import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from scipy import stats
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib
matplotlib.use('Agg')

# Import the logger from src/logging
from logger import logger

def missing_data_summary(data):
    logger.info('Started missing data summary calculation')
    
    # Total missing values per column
    missing_data = data.isnull().sum()
    logger.info(f'Missing data identified in columns: {missing_data[missing_data > 0].index.tolist()}')

    # Filter only columns with missing values greater than 0
    missing_data = missing_data[missing_data > 0]
    
    # Calculate the percentage of missing data
    missing_percentage = (missing_data / len(data)) * 100
    
    # Combine the counts and percentages into a DataFrame
    missing_data = pd.DataFrame({
        'Missing Count': missing_data, 
        'Percentage (%)': missing_percentage
    })
    
    # Sort by percentage of missing data
    missing_data = missing_data.sort_values(by='Percentage (%)', ascending=False)
    
    logger.info('Completed missing data summary calculation')
    return missing_data


def scatter_plot(data, x_col, y_col, hue_col=None):
    logger.info(f'Started scatter plot creation for {x_col} vs {y_col}')
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(data=data, x=x_col, y=y_col, hue=hue_col)
    ax.set_title(f'Scatter Plot of {x_col} vs {y_col}')
    
    logger.info(f'Successfully created scatter plot for {x_col} vs {y_col}')
    plt.tight_layout()
    return fig


def correlation_matrix(data, cols):
    logger.info(f'Started correlation matrix calculation for columns: {cols}')
    
    corr_matrix = data[cols].corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    ax.set_title('Correlation Matrix')
    plt.xticks(rotation=45, ha='right')
    
    logger.info('Completed correlation matrix plot')
    plt.tight_layout()
    return fig


def plot_outliers_boxplot(data, cols):
    logger.info(f'Started outlier boxplot creation for columns: {cols}')
    
    # If a single column is passed as a string, convert it to a list
    if isinstance(cols, str):
        cols = [cols]
    
    # Create subplots
    fig, axes = plt.subplots(1, len(cols), figsize=(12, 4))
    
    # If only one column, 'axes' is not a list, so we need to handle it differently
    if len(cols) == 1:
        axes = [axes]  # Convert single Axes Subplot to a list for consistent indexing

    # Plot the boxplots
    for ax, col in zip(axes, cols):
        sns.boxplot(y=data[col], color='lightblue', ax=ax)
        ax.set_title(f'Box Plot of {col}')
        logger.info(f'Created box plot for column {col}')
    
    plt.tight_layout()
    logger.info('Completed boxplot creation')
    return fig


def cap_all_outliers(data, numerical_columns):
    logger.info(f'Started capping outliers for columns: {numerical_columns}')
    
    for column in numerical_columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap the outliers
        data[column] = data[column].apply(lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x))
        logger.info(f'Outliers capped for {column} with bounds: {lower_bound}, {upper_bound}')
    
    logger.info('Completed capping of outliers')
    return data


def plot_decile_pie_chart(df, column, decile_column='CompetitionDistance', decile_labels=5, title_prefix="mean"):
    logger.info(f'Started decile-based pie chart creation for {column} grouped by {decile_column}')
    
    # Drop rows with missing or invalid data
    df = df.dropna(subset=[column, decile_column]).copy()
    df.loc[:, column] = pd.to_numeric(df[column], errors='coerce')
    df.loc[:, decile_column] = pd.to_numeric(df[decile_column], errors='coerce')
    
    # Adding Decile_rank column to the DataFrame
    df['Decile_rank'] = pd.qcut(df[decile_column], decile_labels, labels=False)

    # Creating a new DataFrame with the decile rank and the chosen column (e.g., Sales or Customers)
    new_df = df[['Decile_rank', column]]
    
    # Grouping by Decile_rank and calculating the mean of the selected column
    a = new_df.groupby('Decile_rank').mean()

    # Ensure the resulting DataFrame has valid sizes
    if a.empty or a[column].isnull().all():
        logger.warning('No valid data to plot in pie chart.')
        return None  # Or handle appropriately

    # Preparing labels and sizes for the pie chart
    labels = a.index.to_list()
    sizes = a[column].to_list()

    # Dynamically generate the 'explode' array to match the number of deciles
    explode = [0.03] * len(sizes)
    explode[0] = 0.1  # Highlight the first slice by exploding it slightly

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['gold', 'yellowgreen', 'purple', 'lightcoral', 'lightskyblue']

    # Plot the pie chart
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, shadow=True, autopct='%.2f', startangle=140)
    ax.set_title(f'A piechart indicating {title_prefix} {column.lower()} in the {decile_labels} {decile_column} decile classes')

    logger.info(f'Completed pie chart for {column} grouped by {decile_column}')
    return fig
