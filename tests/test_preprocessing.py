import pandas as pd

# Import all the functions to be tested
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from preprocessing import missing_data_summary, scatter_plot, correlation_matrix, plot_outliers_boxplot, cap_all_outliers, plot_decile_pie_chart

import pytest
from matplotlib.figure import Figure

@pytest.fixture
def sample_data():
    # Creating a small sample DataFrame for testing purposes
    data = {
        'Sales': [200, 300, 400, 500, None],
        'Customers': [20, 30, 40, 50, None],
        'CompetitionDistance': [100, 200, 300, 400, 500],
        'Promotion': [1, 0, 1, 1, 0]
    }
    return pd.DataFrame(data)

def test_missing_data_summary(sample_data):
    # Check if missing data summary works correctly
    result = missing_data_summary(sample_data)
    assert isinstance(result, pd.DataFrame)
    assert 'Sales' in result.index
    assert 'Customers' in result.index
    assert result.loc['Sales', 'Missing Count'] == 1
    assert result.loc['Customers', 'Missing Count'] == 1

def test_scatter_plot(sample_data):
    # Check if scatter plot returns a matplotlib Figure object
    fig = scatter_plot(sample_data, 'Sales', 'Customers')
    assert isinstance(fig, Figure)

def test_correlation_matrix(sample_data):
    # Check if correlation matrix plot returns a matplotlib Figure object
    fig = correlation_matrix(sample_data, ['Sales', 'Customers', 'CompetitionDistance'])
    assert isinstance(fig, Figure)

def test_plot_outliers_boxplot(sample_data):
    # Check if box plot function returns a matplotlib Figure object
    fig = plot_outliers_boxplot(sample_data, ['Sales', 'Customers'])
    assert isinstance(fig, Figure)

def test_cap_all_outliers(sample_data):
    # Modify the sample data to include outliers
    sample_data.loc[0, 'Sales'] = 1000  # Introduce an outlier
    
    # Run the outlier capping function
    result = cap_all_outliers(sample_data, ['Sales'])
    
    # Ensure outliers are capped
    assert result['Sales'].max() <= sample_data['Sales'].quantile(0.75) + 1.5 * (sample_data['Sales'].quantile(0.75) - sample_data['Sales'].quantile(0.25))

def test_plot_decile_pie_chart(sample_data):
    # Check if the decile-based pie chart returns a matplotlib Figure object
    fig = plot_decile_pie_chart(sample_data, 'Sales', decile_column='CompetitionDistance')
    assert fig is not None
    assert isinstance(fig, Figure)

if __name__ == "__main__":
    pytest.main()
