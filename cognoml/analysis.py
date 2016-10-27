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

        self.X = X
        self.obs_df = y
        self.y = y.values
        self.pipeline = pipeline
        self.test_size = test_size
        self.x_train, self.x_test, self.y_train, self.y_test = self.test_train_split()
        self.json_sanitize = json_sanitize

    def test_train_split(self):
        x = self.X
        y = self.obs_df
        test_size = self.test_size
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=0, stratify=y)
        return x_train, x_test, y_train, y_test


    def fit(self):
        x_train = self.x_train
        y_train = self.y_train
        pipeline = self.pipeline
        pipeline.fit(X=x_train, y=y_train)

    def predict(self):
        pipeline = self.pipeline
        x = self.X
        predict_df = pd.DataFrame({'sample_id': x.index, 'predicted_status': pipeline.predict(x)})
        if hasattr(pipeline, 'decision_function'):
            predict_df['predicted_score'] = pipeline.decision_function(x)
        if hasattr(pipeline, 'predict_proba'):
            predict_df['predicted_prob'] = pipeline.predict_proba(x)[:, 1]
        return predict_df

    def get_results(self):
        pipeline = self.pipeline
        json_sanitize = self.json_sanitize
        results = collections.OrderedDict()
        x_test = self.x_test
        x = self.X
        obs_df = self.obs_df
        obs_df = pd.DataFrame({'sample_id': obs_df.index, 'status': obs_df.values})
        predict_df = self.predict()
        obs_df['testing'] = obs_df['sample_id'].isin(x_test.index).astype(int)
        obs_df = obs_df.merge(predict_df, how='right', sort=True)
        obs_df['selected'] = obs_df['sample_id'].isin(obs_df['sample_id']).astype(int)
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
        dimensions['positive_prevalence'] = obs_df.status.mean()
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
        if json_sanitize:
            results = utils.make_json_serializable(results)
        return results


def classify(sample_id, mutation_status, data_version, json_sanitize=False, **kwargs):
    """
    Perform an analysis.
    
    Parameters
    ----------
    sample_id : list
        Sample IDs of the observations.
    mutation_status : list
        Mutation status (0 or 1) of each sample.
    data_version : int
        Integer with the figshare data version.
    json_sanitize : bool
        Whether to make results JSON-serializable. If `True` DataFrames are
        converted to DataTables format.

    Returns
    -------
    results : dict
        An object of results. See `data/api/hippo-output-schema.json`
        for JSON schema.
    """
    results = collections.OrderedDict()
    
    obs_df = pd.DataFrame.from_items([
        ('sample_id', sample_id),
        ('status', mutation_status),
    ])
    
    X_whole = read_data(version=data_version)
    X = X_whole.loc[obs_df.sample_id, :]
    y = obs_df.status
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=0, stratify=y)
    obs_df['testing'] = obs_df.sample_id.isin(X_test.index).astype(int)
    
    grid_search.fit(X=X_train, y=y_train)
    
    predict_df = pd.DataFrame.from_items([
        ('sample_id', X_whole.index),
        ('predicted_status', grid_search.predict(X_whole)),
    ])
    if hasattr(grid_search, 'decision_function'):
        predict_df['predicted_score'] = grid_search.decision_function(X_whole)
    if hasattr(grid_search, 'predict_proba'):
        predict_df['predicted_prob'] = grid_search.predict_proba(X_whole)[:, 1]
    
    # obs_df switches to containing non-selected samples
    obs_df = obs_df.merge(predict_df, how='right', sort=True)
    obs_df['selected'] = obs_df.sample_id.isin(sample_id).astype(int)
    for column in 'status', 'testing', 'selected':
        obs_df[column] = obs_df[column].fillna(-1).astype(int)
    obs_train_df = obs_df.query("testing == 0")
    obs_test_df = obs_df.query("testing == 1")

    #y_pred_train = obs_df.query("testing == 0").predicted_score
    #y_pred_test = obs_df.query("testing == 1").predicted_score

    dimensions = collections.OrderedDict()
    dimensions['observations_selected'] = sum(obs_df.selected == 1)
    dimensions['observations_unselected'] = sum(obs_df.selected == 0)
    dimensions['features'] = len(X.columns)
    dimensions['positives'] = sum(obs_df.status == 1)
    dimensions['negatives'] = sum(obs_df.status == 0)
    dimensions['positive_prevalence'] = y.mean()
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
    performance['cv'] = {'auroc': grid_search.best_score_}
    results['performance'] = performance
    
    gs = collections.OrderedDict()
    gs['cv_scores'] = utils.cv_results_to_df(grid_search.cv_results_)
    results['grid_search'] = gs
    
    results['model'] = utils.model_info(grid_search.best_estimator_.steps[-1][1])

    feature_df = utils.get_feature_df(grid_search, X.columns)
    results['model']['features'] = feature_df

    results['observations'] = obs_df
    
    if json_sanitize:
        results = utils.make_json_serializable(results)
    
    return results
