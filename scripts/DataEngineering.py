import refinitiv.data as rd
from refinitiv.data.content import news
from refinitiv.data.content import esg
import pandas as pd
import datetime
from datetime import datetime
from dateutil.relativedelta import relativedelta


class DataEngineering:
    def __init__(self, assets, start, end, window) -> None:
        self.assets = assets
        self.start = start
        self.end = end
        self.window = window

    def run(self):
        stocks_prices = self.get_prices(self.assets).dropna()
        benchmark_prices = self.get_prices('.SPX').dropna()
        industry = self.get_industry()
        ratios = self.get_daily_ratios()
        financials = self.get_financials()
        ranking = self.get_ranking_estimates()
        trading_activity = self.get_trading_activity()
        num_analyst = self.get_analyst_coverage()
        news = self.get_news()
        return {'prices': {'stocks_prices': stocks_prices, 'benchmark_prices': benchmark_prices},
                'raw_data': {'industry': industry, 'ratios': ratios, 'financials': financials, 'ranking': ranking,
                             'trading_activity': trading_activity, 'num_analyst': num_analyst, 'news': news}}

    def get_prices(assets, self):
        start = self.start - relativedelta(days=1.5*self.window)
        return rd.get_history(assets, fields=['TRDPRC_1'], start=start.strftime("%Y-%m-%d"),
                              end=self.end.strftime("%Y-%m-%d"))

    def get_daily_ratios(self):
        print("Getting Daily Ratios")
        daily_ratios = rd.get_history(self.assets,
                                      fields=["TR.PriceToBVPerShare", "TR.PriceToSalesPerShare",
                                              "TR.EVToEBITDA", "TR.EVToSales", "TR.TotalDebtToEV", "TR.TotalDebtToEBITDA"],
                                      start=self.start, end=self.end)
        return self.reshape_data(daily_ratios)

    def get_financials(self):
        print("Getting Financials")
        financials = rd.get_history(self.assets,
                                    fields=["TR.F.MktCap", "TR.F.ReturnAvgTotEqPctTTM", "TR.F.IncAftTaxMargPctTTM",
                                            "TR.F.GrossProfMarg", "TR.F.NetIncAfterMinIntr", "TR.F.TotCap",
                                            "TR.F.OpMargPctTTM", "TR.F.ReturnCapEmployedPctTTM", "TR.F.NetCashFlowOp",
                                            "TR.F.LeveredFOCF", "TR.F.TotRevenue", "TR.F.RevGoodsSrvc3YrCAGR",
                                            "TR.F.NetPPEPctofTotAssets", "TR.F.TotAssets", "TR.F.SGA", "TR.F.CurrRatio",
                                            "TR.F.WkgCaptoTotAssets", "TR.F.TotShHoldEq", "TR.F.TotDebtPctofTotAssets",
                                            "TR.F.DebtTot", "TR.F.NetDebtPctofNetBookValue", "TR.F.NetDebttoTotCap",
                                            "TR.F.NetDebtPerShr"
                                            ],
                                    start=self.start, end=self.end)

        return self.reshape_data(financials)

    def get_trading_activity(self):
        print("Getting Trading Activities")
        trading_activity = rd.get_history(self.assets, fields=['BID', 'ASK', 'HIGH_1', 'LOW_1', 'ACVOL_UNS',
                                                               'TRNOVR_UNS', 'BLKCOUNT', 'BLKVOLUM', 'NUM_MOVES'],
                                          start=self.start, end=self.end)
        trading_activity = self.reshape_data(trading_activity)
        trading_activity['Spread'] = trading_activity['BID'] - \
            trading_activity['ASK']
        trading_activity['Range'] = trading_activity['HIGH_1'] - \
            trading_activity['LOW_1']

        return trading_activity

    def get_ranking_estimates(self):
        print("Getting Ranking Estimates")
        estimates = rd.get_history(self.assets,
                                   fields=[
                                       'TR.CombinedAlphaRegionRank',
                                       'TR.CombinedAlphaCountryRank',
                                       'TR.CombinedAlphaSectorRank',
                                       'TR.CombinedAlphaIndustryRank',
                                   ],
                                   start=self.start, end=self.end)

        return self.reshape_data(estimates)

    def get_value_chain_score(self):
        print("Getting Value chain scores")
        value_chain_score = rd.get_data(self.assets,
                                        fields=['TR.SCRelationship', 'TR.SCRelationshipConfidenceScore',
                                                'TR.SCRelationshipUpdateDate'],
                                        parameters={
                                            'SDate': self.start.strftime("%Y-%m-%d"), 'EDate': self.end.strftime("%Y-%m-%d")})
        value_chain_score.sort_values(
            by=['Value Chains Relationship Update Date'], ascending=True, inplace=True)
        value_chain_score.rename(
            columns={"Value Chains Relationship Update Date": "Date"}, inplace=True)
        value_chain_score_reshaped = pd.pivot_table(
            value_chain_score,  index=['Instrument', 'Date']).reset_index()

        return value_chain_score_reshaped

    def get_news(self):
        print('Getting News Data')
        newsdf = pd.DataFrame()
        i = 0
        for asset in self.assets:
            i += 1
            print(i, asset, newsdf.shape)
            end = self.end
            end_init = " "
            while end >= self.start and end != end_init:
                try:
                    end_init = end
                    response = news.headlines.Definition(
                        query=f"{asset} and L:EN", count=10000, date_to=end).get_data().data.df
                    response.insert(0, 'Asset', asset)
                    newsdf = pd.concat([newsdf, response])
                    end = datetime.date(response.index.min())
                except:
                    continue
        newsdf = newsdf.drop_duplicates(subset=['Asset', 'headline']).reset_index(
            drop=True).rename(columns={"Asset": "Instrument"})
        return newsdf

    def get_analyst_coverage(self):
        print("Getting Number of Estimates")
        num_analysts = rd.get_data(self.assets, fields=['TR.NumberOfAnalysts', 'TR.NumberOfAnalysts.date'], parameters={
            'SDate': self.start.strftime("%Y-%m-%d"), 'EDate': self.end.strftime("%Y-%m-%d")})
        return num_analysts

    def get_industry(self):
        print("Getting industry name")
        return rd.get_data(self.assets, 'TR.TRBCIndustryGroup')

    def reshape_data(self, df):
        df = df.stack(
            level=0).reset_index().rename(columns={"level_1": "Instrument"})
        df_reshaped = pd.pivot_table(
            df,  index=['Instrument', 'Date']).reset_index()
        return df_reshaped
