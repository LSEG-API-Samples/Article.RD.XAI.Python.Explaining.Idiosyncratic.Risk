import pandas as pd
import tensorflow as tf
import os
import refinitiv.data as rd

from datetime import date
from dateutil.relativedelta import relativedelta
from FeatureEngineering import FeatureEngineering
from DataEngineering import DataEngineering
from mna import XAI

tf.keras.backend.set_floatx('float64')

os.environ["RD_LIB_CONFIG_PATH"] = "Configuration"
rd.open_session()

end = date.today()
start = pd.to_datetime(end - relativedelta(months=15))
beta_window = 90
study_scope = 'stock'

assets = rd.get_data('.OEXA', 'TR.IndexConstituentRIC')[
    'Constituent RIC'].to_list()

de = DataEngineering(assets, start, end, beta_window).run()
label_dfs = de['prices']
features_dfs = de['raw_data']

fe = FeatureEngineering(start, end, beta_window, study_scope)
studies = fe.run(label_dfs, features_dfs)

for study_name, study_feature_set in studies.items():
    XAI(study_name, study_feature_set['df_x'], study_feature_set['df_y']).fit()
