study_features = [
    'Enterprise Value To EBITDA (Daily Time Series Ratio)',
    'Enterprise Value To Sales (Daily Time Series Ratio)',
    'Price To Book Value Per Share (Daily Time Series Ratio)',
    'Price To Sales Per Share (Daily Time Series Ratio)',
    'Total Debt To EBITDA (Daily Time Series Ratio)',
    'Total Debt To Enterprise Value (Daily Time Series Ratio)',
    'Combined Alpha Model Country Rank',
    'Combined Alpha Model Industry Rank',
    'ACVOL_UNS', 'ASK', 'BID',
    'BLKCOUNT', 'BLKVOLUM',
    'HIGH_1', 'LOW_1',
    'NUM_MOVES',
    'TRNOVR_UNS',
    'Spread',
    'Range',
    'Number of Analysts',
    'positive',
    'negative', 'neutral',
    'news_count',
    'industry'
]

drop_features = ['Current Ratio', 'Debt - Total', 'Free Cash Flow',
                 'Gross Profit Margin - %', 'Income after Tax Margin - %, TTM',
                 'Market Capitalization', 'Net Cash Flow from Operating Activities',
                 'Net Debt Percentage of Net Book Value', 'Net Debt per Share',
                 'Net Debt to Total Capital', 'Net Income after Minority Interest',
                 'Operating Margin - %, TTM', 'PPE - Net Percentage of Total Assets',
                 'Return on Average Total Equity - %, TTM',
                 'Return on Capital Employed - %, TTM',
                 'Revenue from Business Activities - Total',
                 'Revenue from Goods & Services, 3 Yr CAGR',
                 'Selling General & Administrative Expenses', 'Total Assets',
                 'Total Capital', 'Total Debt Percentage of Total Assets',
                 'Total Shareholders Equity incl Minority Intr & Hybrid Debt',
                 'Working Capital to Total Assets'
                 ]

xai_logs_path = './xai_logs/'
