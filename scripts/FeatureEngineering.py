import pandas as pd
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tqdm import tqdm
from LabelBuilder import LabelBuilder
from functools import reduce
from utilities import drop_features, study_features


class FeatureEngineering:

    def __init__(self, start, end, beta_window, scope) -> None:
        self.start = start
        self.end = end
        self.scope = scope
        self.lb = LabelBuilder(beta_window)

    def run(self, label_dfs, features_dfs):
        labels = self.get_labels(
            label_dfs['stocks_prices'], label_dfs['benchmark_prices'])
        news_with_sent = self.get_sentiments(features_dfs['news'])
        features_dfs['news'] = news_with_sent
        dataset_raw = self.build_dataset(labels, features_dfs)
        dataset_processed = self.preprocess_data(dataset_raw)
        studies = self.finalise_feature_set(
            dataset_processed, study_features, self.scope)

        return studies

    def finalise_feature_set(self, feature_set, study_features, scope):
        studies = {}
        feature_set.drop(columns=drop_features, inplace=True)
        feature_set['Date'] = pd.to_datetime(
            feature_set['Date']).dt.date.astype('datetime64')
        feature_set = feature_set.sort_values(by='Date')
        feature_set.dropna(inplace=True)

        if scope == 'all':
            df_y = feature_set['rank_label']
            df_x = feature_set[study_features]
            studies[scope] = {"df_x": df_x, "df_y": df_y}

        elif scope == 'industry':
            study_features.remove('industry')
            industries = [
                col for col in feature_set.columns if 'Industry_' in col]
            for industry in industries:
                df_group = feature_set[feature_set[industry] == 1]
                df_y = df_group['rank_label']
                df_x = df_group[study_features]
                studies[industry] = {"df_x": df_x, "df_y": df_y}
        elif scope == 'stock':
            study_features.remove('industry')
            feature_set_group = feature_set.groupby(by='Instrument')
            for name, df_group in feature_set_group:
                df_y = df_group['rank_label']
                df_x = df_group[study_features]
                studies[name] = {"df_x": df_x, "df_y": df_y}
        else:
            print('Invalid Scope')

        return studies

    def get_labels(self, stocks_prices, benchmark_prices):
        print('Getting Labels')
        labels = self.lb.get_ranked_residuals(
            stocks_prices, benchmark_prices)

        return labels

    def get_sentiments(self, newsdf):
        newsdf_with_finbert = self.get_sentiments_finbert(newsdf)
        sent_B = self.get_sentiment_bart(newsdf)
        newsdf_with_sents = pd.concat(
            [newsdf_with_finbert, sent_B], axis=1, join="inner")
        return self.build_features_on_sentiment(newsdf_with_sents)

    def get_sentiments_finbert(self, newsdf):
        print('Calculating news sentiments')
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained(
            "ProsusAI/finbert")
        label_list = ['positive', 'negative', 'neutral']
        sents = []
        for headline in newsdf['headline']:
            tokenized = tokenizer(headline, return_tensors="pt")
            outputs = model(**tokenized)
            sent = label_list[torch.argmax(outputs[0])]
            sents.append(sent)
        newsdf['sentiment'] = sents

        return newsdf

    def get_sentiment_bart(self, newsdf):
        classifier = pipeline("zero-shot-classification",
                              model="facebook/bart-large-mnli", device=torch.device("mps"))
        labels = ['positive', 'negative', 'neutral']

        sentiments = {'versionCreated': [], 'headline': [],
                      'positive': [], 'negative': [], 'neutral': []}

        for index, news_item in tqdm(newsdf.iterrows(), total=newsdf.shape[0]):
            sent = classifier(news_item['headline'], labels)
            pred_labels = sent["labels"]
            scores = sent["scores"]
            for label, score in zip(pred_labels, scores):
                sentiments[label].append(score)
            sentiments['headline'].append(news_item['headline'])
            sentiments['versionCreated'].append(news_item['versionCreated'])

        return pd.DataFrame(sentiments)

    def build_features_on_sentiment(self, newsdf_with_sents):
        newsdf_with_sents = newsdf_with_sents.loc[:,
                                                  ~newsdf_with_sents.columns.duplicated()]
        newsdf_with_sents = newsdf_with_sents.rename(
            columns={'versionCreated': 'Date'})
        newsdf_with_sents['Date'] = pd.to_datetime(
            newsdf_with_sents['Date']).dt.date.astype('datetime64')
        newsdf_with_sents['sent_num'] = np.where(newsdf_with_sents['sentiment'] == 'positive', 1, np.where(
            newsdf_with_sents['sentiment'] == 'negative', -1, 0))
        news_count = newsdf_with_sents.groupby(['Instrument', 'Date'])[
            'sentiment'].count().reset_index()['sentiment'].to_list()
        newsdf_with_sents = newsdf_with_sents.groupby(
            ['Instrument', 'Date']).sum().reset_index()
        newsdf_with_sents['news_count'] = news_count

        return newsdf_with_sents

    def build_dataset(self, labels, features):
        dfs = [labels] + list(features.values())[1:]
        new_dfs = []
        for df in dfs:
            df['Instrument'] = df['Instrument'].str.upper()
            df['Date'] = pd.to_datetime(
                df['Date']).dt.date.astype('datetime64')
            duplicated_mask = df.duplicated(
                subset=["Date", "Instrument"], keep="first")
            df = df[(df['Date'] < self.end) & (df['Date'] > self.start)]
            df = df[~duplicated_mask]
            new_dfs.append(df)
        merged_df = reduce(lambda left, right: pd.merge(
            left, right, on=["Date", "Instrument"], how="outer"), new_dfs).set_index('Date')
        merged_df = merged_df.reset_index().sort_values(
            by=['Instrument', 'Date'], ascending=[True, True]).set_index('Instrument')
        merged_df = merged_df.groupby(merged_df.index).fillna(
            method='ffill').reset_index().set_index('Date')

        return merged_df.reset_index().merge(list(features.values())[0])

    def preprocess_data(self, dataset):
        dataset['TRBC Industry Group Name'] = dataset['TRBC Industry Group Name'].astype(
            'category')
        industry_num = dataset['TRBC Industry Group Name'].cat.codes + 1
        dataset.insert(53, 'industry', industry_num)
        num_features = dataset.iloc[:, 3:54]

        scaler = MinMaxScaler()
        encoder = OneHotEncoder()
        scaled_df = pd.DataFrame(scaler.fit_transform(
            num_features), columns=num_features.columns)
        encoded_data = encoder.fit_transform(
            dataset['TRBC Industry Group Name'].array.reshape(-1, 1)).toarray()
        encoded_df = pd.DataFrame(
            encoded_data, columns=encoder.get_feature_names_out(['Industry']))
        dataset = pd.concat(
            [dataset[['Date', 'Instrument', 'rank_label']], scaled_df, encoded_df], axis=1).set_index('Date', drop=True)

        return dataset
