import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch

# Import all the functions to be tested
from src.preprocessing import missing_data_summary, scatter_plot, correlation_matrix, plot_outliers_boxplot, cap_all_outliers, plot_decile_pie_chart

class TestFunctions(unittest.TestCase):

    @patch('src.logger.logger')
    def test_missing_data_summary(self, mock_logger):
        data = pd.DataFrame({
            'A': [1, 2, None, 4],
            'B': [None, 2, 3, 4],
            'C': [1, None, None, 4],
            'D': [1, 2, 3, 4]  # No missing data
        })
        expected_output = pd.DataFrame({
            'Missing Count': [2, 2, 1],
            'Percentage (%)': [50.0, 50.0, 25.0]
        }, index=['B', 'C', 'A']).sort_values(by='Percentage (%)', ascending=False)

        result = missing_data_summary(data)
        pd.testing.assert_frame_equal(result, expected_output)

        mock_logger.info.assert_any_call('Started missing data summary calculation')
        mock_logger.info.assert_any_call('Completed missing data summary calculation')

    @patch('src.logger.logger')
    def test_scatter_plot(self, mock_logger):
        data = pd.DataFrame({
            'A': np.random.rand(10),
            'B': np.random.rand(10),
            'C': ['cat1', 'cat2'] * 5
        })
        fig = scatter_plot(data, 'A', 'B', 'C')

        self.assertIsInstance(fig, plt.Figure)
        mock_logger.info.assert_any_call('Started scatter plot creation for A vs B')
        mock_logger.info.assert_any_call('Successfully created scatter plot for A vs B')

    @patch('src.logger.logger')
    def test_correlation_matrix(self, mock_logger):
        data = pd.DataFrame({
            'A': np.random.rand(10),
            'B': np.random.rand(10),
            'C': np.random.rand(10)
        })
        fig = correlation_matrix(data, ['A', 'B', 'C'])

        self.assertIsInstance(fig, plt.Figure)
        mock_logger.info.assert_any_call('Started correlation matrix calculation for columns: [\'A\', \'B\', \'C\']')
        mock_logger.info.assert_any_call('Completed correlation matrix plot')

    @patch('.src.logger.logger')
    def test_plot_outliers_boxplot(self, mock_logger):
        data = pd.DataFrame({
            'A': np.random.randn(100),
            'B': np.random.randn(100)
        })
        fig = plot_outliers_boxplot(data, ['A', 'B'])

        self.assertIsInstance(fig, plt.Figure)
        mock_logger.info.assert_any_call('Started outlier boxplot creation for columns: [\'A\', \'B\']')
        mock_logger.info.assert_any_call('Completed boxplot creation')

    @patch('src.logger.logger')
    def test_cap_all_outliers(self, mock_logger):
        data = pd.DataFrame({
            'A': [10, 12, 14, 1000, 15, 13],
            'B': [-999, 0, 0.5, 1, 2, 3]
        })
        expected_output = pd.DataFrame({
            'A': [10, 12, 14, 15.0, 15, 13],  # 1000 capped to 15.0
            'B': [0.0, 0, 0.5, 1, 2, 3]     # -999 capped to 0.0
        })

        result = cap_all_outliers(data, ['A', 'B'])
        pd.testing.assert_frame_equal(result, expected_output)

        mock_logger.info.assert_any_call('Started capping outliers for columns: [\'A\', \'B\']')
        mock_logger.info.assert_any_call('Completed capping of outliers')

    @patch('src.logger.logger')
    def test_plot_decile_pie_chart(self, mock_logger):
        data = pd.DataFrame({
            'CompetitionDistance': [500, 1000, 1500, 2000, 2500],
            'Sales': [100, 150, 200, 250, 300]
        })
        fig = plot_decile_pie_chart(data, 'Sales', 'CompetitionDistance', decile_labels=5)

        self.assertIsInstance(fig, plt.Figure)
        mock_logger.info.assert_any_call('Started decile-based pie chart creation for Sales grouped by CompetitionDistance')
        mock_logger.info.assert_any_call('Completed pie chart for Sales grouped by CompetitionDistance')

if __name__ == '__main__':
    unittest.main()
