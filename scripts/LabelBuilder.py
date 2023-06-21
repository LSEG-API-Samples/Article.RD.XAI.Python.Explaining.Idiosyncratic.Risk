import pandas as pd
import statsmodels.api as sm
from functools import reduce


class LabelBuilder:
    def __init__(self, beta_window) -> None:
        self.beta_window = beta_window

    def get_ranked_residuals(self, stocks_prices, benchmark_prices):
        residuals = pd.DataFrame()
        for stock in stocks_prices.columns:
            print(stock)
            stock_prices = stocks_prices[stock]
            residuals_stock = self.calculate_residuals(
                stock_prices, benchmark_prices)
            residuals[stock] = residuals_stock['Residuals']
        labels = self.rank_stock_by_risk(residuals)
        return labels

    def calculate_residuals(self, stock_prices, benchmark_prices):

        stock_returns = stock_prices.pct_change().dropna()
        benchmark_returns = benchmark_prices.pct_change().dropna()
        residuals = {}
        for i in range(self.beta_window, len(stock_returns)+1):
            stock_window = stock_returns[i -
                                         self.beta_window:i].values.reshape(-1, 1)
            benchmark_window = benchmark_returns[i -
                                                 self.beta_window:i].values.reshape(-1, 1)
            X = sm.add_constant(benchmark_window)
            model = sm.OLS(stock_window.astype(float), X.astype(float))
            results = model.fit()
            residuals[stock_returns.index[i-1]
                      ] = sum(abs(number) for number in results.resid)
        resids = pd.DataFrame(residuals, index=['Residuals']).T
        return resids

    def rank_stock_by_risk(self, residuals):
        ranked_resids = residuals.rank(pct=True, axis=1)
        ranked_resids_normed = (ranked_resids-0.5)*3.46
        ranked_resids_normed = pd.melt(ranked_resids_normed.reset_index(), id_vars=[
            'index'], value_vars=ranked_resids_normed.columns, var_name='Instrument', value_name='rank_label')
        ranked_resids_normed.rename(columns={'index': 'Date'}, inplace=True)
        return ranked_resids_normed
