import collections

import pandas as pd
from sklearn.model_selection import train_test_split

from cognoml import utils
from cognoml.classifiers.logistic_regression import grid_search

class CognomlClassifier:
    """
    Class to handle all operations related to the Cognoml Classifier
    """

    def __init__(self, X, y, pipeline=grid_search, test_size=0.1, json_sanitize=True):
        """

        Parameters
        ----------
        X: Pandas data frame
            Expressions data frame, [sample_id, feature set]
        y: Pandas data frame
            Mutations data frame [sample_id, boolean (1 or 0) on mutation]
        routine: function
            ML model fitting pipeline, of form function(X, y)
        test_size: float
            % of total sample to be used in testing data. Training size = 1-test_size
        json_sanitize: bool
            Whether to make results JSON-serializable. If `True` DataFrames are
            converted to DataTables format.
        """

        self.X_whole = X
        self.X = utils.filter_data_by_mutation(X, y)
        self.obs_df = y
        self.sample_id = self.obs_df.index
        self.y = y.values
        self.pipeline = pipeline
        self.test_size = test_size
        self.x_train, self.x_test, self.y_train, self.y_test = self.test_train_split()
        self.json_sanitize = json_sanitize

    def test_train_split(self):
        """
        Internal wrapper for scikit-learn's test/train split
        Performs stratified test/train split of Cognoml data sets
        Stratify option set to consume the mutations data by default

        Returns
        -------
        x_train: Pandas data frame
            Training set of expressions data frame (feature space)
        x_test: Pandas data frame
            Test set of expressions data frame
        y_train: Pandas data frame
            Training set of mutations data
        y_test: Pandas data frame
            Test set of mutations data
        """
        x = self.X
        y = self.y
        test_size = self.test_size
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=0, stratify=y)
        return x_train, x_test, y_train, y_test


    def fit(self):
        """
        Internal wrapper for scikit-learn's fit method on a custom data pipeline
        Fits custom data pipeline using internal training data sets created
        by test_train_split

        """
        x_train = self.x_train
        y_train = self.y_train
        pipeline = self.pipeline
        try:
            pipeline.fit(X=x_train, y=y_train)
        except AttributeError:
            print("Pipeline {} does not have a fit method".format(pipeline))

    def predict(self):
        """
        Internal wrapper for scikit-learn's predict method with custom data pipeline

        Returns
        -------
        predict_df: Pandas data frame
            Mutation predictions for entire feature dataframe
        """
        pipeline = self.pipeline
        x = self.X_whole
        try:
            predict_df = pd.DataFrame(collections.OrderedDict((('sample_id', x.index),
                                                               ('predicted_status', pipeline.predict(x)))))
        except AttributeError:
            raise AttributeError("Pipeline {} does not have a predict method".format(pipeline))
        if hasattr(pipeline, 'decision_function'):
            predict_df['predicted_score'] = pipeline.decision_function(x)
        if hasattr(pipeline, 'predict_proba'):
            predict_df['predicted_prob'] = pipeline.predict_proba(x)[:, 1]
        return predict_df

    def get_results(self):
        pipeline = self.pipeline
        results = collections.OrderedDict()
        x_test = self.x_test
        x = self.X
        obs_df = self.obs_df
        obs_df = pd.DataFrame(collections.OrderedDict((('sample_id', obs_df.index), ('status', obs_df.values))))
        predict_df = self.predict()
        obs_df['testing'] = obs_df['sample_id'].isin(x_test.index).astype(int)
        obs_df = obs_df.merge(predict_df, how='right', sort=True)
        obs_df['selected'] = obs_df['sample_id'].isin(self.sample_id).astype(int)
        for column in 'status', 'testing', 'selected':
            obs_df[column] = obs_df[column].fillna(-1).astype(int)
        obs_train_df = obs_df.query("testing == 0")
        obs_test_df = obs_df.query("testing == 1")
        dimensions = collections.OrderedDict()
        dimensions['observations_selected'] = sum(obs_df.selected == 1)
        dimensions['observations_unselected'] = sum(obs_df.selected == 0)
        dimensions['features'] = len(x.columns)
        dimensions['positives'] = sum(obs_df.status == 1)
        dimensions['negatives'] = sum(obs_df.status == 0)
        dimensions['positive_prevalence'] = obs_df.query("selected == 1").status.mean()
        dimensions['training_observations'] = len(obs_train_df)
        dimensions['testing_observations'] = len(obs_test_df)
        results['dimensions'] = dimensions
        performance = collections.OrderedDict()
        for part, df in ('training', obs_train_df), ('testing', obs_test_df):
            y_true = df.status
            y_pred = df.predicted_status
            metrics = utils.class_metrics(y_true, y_pred)
            metrics.update(utils.threshold_metrics(y_true, y_pred))
            performance[part] = metrics
        performance['cv'] = {'auroc': pipeline.best_score_}
        results['performance'] = performance
        gs = collections.OrderedDict()
        gs['cv_scores'] = utils.cv_results_to_df(pipeline.cv_results_)
        results['grid_search'] = gs
        # CHECK BELOW VERY THOROUGHLY
        results['model'] = utils.model_info(pipeline.best_estimator_.steps[-1][1])
        feature_df = utils.get_feature_df(pipeline, x.columns)
        results['model']['features'] = feature_df
        results['observations'] = obs_df
        if self.json_sanitize:
            results = utils.make_json_serializable(results)
        return results
