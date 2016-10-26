import os
from urllib.request import urlretrieve
import json
import requests
import pandas as pd


class CognomlData:
    """Class to manage downloading, cleaning, and filtering data for the Cognoml Machine Learning project."""

    def __init__(self, article_id=3487685, figshare_url='https://api.figshare.com/v2/articles/{}/versions',
                 directory='download', mutation_file='mutation-matrix.tsv.bz2',
                 expressions_file='expression-matrix.tsv.bz2', version=None):
        """
        Attributes:
            article_id (int): figshare article id of data, defaults to 3487685
            figshare_url (str): full figshare URL of data, defaults to 'https://api.figshare.com/v2/articles/{}/versions'
            directory (str): local directory to be created for downloading data, defaults to 'download'
            mutation_file (str): name of zipped tsv file containing mutation data, defaults to 'mutation-matrix.tsv.bz2'
            expressions_file (str): name of zipped tsv file containing expressions data, defaults to
                'expression-matrix.tsv.bz2'
            version (str): name of data version to be used, defaults to None

        """
        self._article_id = article_id
        self._figshare_url = figshare_url
        self._directory = directory
        self._version_to_url = self.get_article_version()
        self._version = self.get_version(version)
        self._download_path = os.path.join(self._directory, 'v{}'.format(self._version))
        self._mutation_file = mutation_file
        self._expressions_file = expressions_file
        self._mutations_path = os.path.join(self._download_path, self._mutation_file)
        self._expressions_path = os.path.join(self._download_path, self._expressions_file)

    def get_article_version(self):
        """
        Get version_to_url dictionary for a figshare article.

        Returns
        -------
        version_to_url: dict{'version': 'url'}
            Dict with key=version and value=full figshare URL
        """
        article_id = self._article_id
        figshare_url = self._figshare_url
        url = figshare_url.format(article_id)
        response = requests.get(url)
        version_to_url = {d['version']: d['url'] for d in response.json()}
        return version_to_url

    def get_version(self, init_version):
        """
        Get latest version if version = None

        Parameters
        ----------
        init_version: str

        Returns
        -------
        version: str

        """
        version_to_url = self._version_to_url
        if init_version is None:
            version = max(version_to_url.keys())
            return version
        else:
            return init_version

    def download_files(self):
        """
        Download files for figshare article_id and version. Creates a
        version-specific subdirectory in `directory` and downloads all files from
        the figshare article into this subdirectory.

        Returns
        -------
        download_path: str
            The version-specific directory corresponding to the downloaded data.
        """
        version_to_url = self._version_to_url
        version = self._version
        if version is None:
            version = max(version_to_url.keys())
        url = version_to_url[version]
        response = requests.get(url)
        article = response.json()
        download_path = self._download_path

        if not os.path.exists(download_path):
            os.mkdir(download_path)

        path = os.path.join(download_path, 'info.json')
        with open(path, 'w') as write_file:
            json.dump(article, write_file, indent=2, ensure_ascii=False, sort_keys=True)

        # Download the files specified by the metadata
        for file_info in article['files']:
            name = file_info['name']
            path = os.path.join(download_path, name)
            if os.path.exists(path):
                print('{} already exists, checking next file'.format(path))
                continue
            url = file_info['download_url']
            print('Downloading {} to `{}`'.format(url, name))
            urlretrieve(url, path)

        return download_path

    def get_df_from_table(self, tsv_file):
        """
        Checks for local pickle file to load data frame, if it does not exist loads local tsv file to data frame,
        then writes to pickle file for future use.

        Parameters
        ----------
        tsv_file: str
            Filename of tsv file containing mutation or expression data

        Returns
        -------
        df: Pandas Data frame
        """
        expr_file = self._expressions_file
        mut_file = self._mutation_file
        pickle_dict = {expr_file: 'mutation.p',
                       mut_file: 'expression.p'}
        try:
            pickle_file = pickle_dict[tsv_file]
        except KeyError:
            raise KeyError('{} not expected input, choose either {} or {}'.format(tsv_file, expr_file, mut_file))
        download_path = self._download_path
        data_path = os.path.join(download_path, tsv_file)
        pickle_path = os.path.join(download_path, pickle_file)
        if os.path.exists(pickle_path):
            df = pd.read_pickle(pickle_path)
            print('Reading data frame from cached pickle file {}'.format(pickle_path))
        else:
            if not os.path.exists(data_path):
                raise IOError('bz2 file does not exist, try running download_files()')
            print('Reading data frame from file {}'.format(data_path))
            df = pd.read_table(data_path, index_col=0)
            df.to_pickle(pickle_path)
        return df

    @staticmethod
    def filter_data_by_mutation(mutation, expr_df, mut_df):
        """
        Filters total data sets to include only data for a single mutation

        Parameters
        ----------
        mutation: String
            Mutation number as string, for example '1'

        expr_df: Pandas Data frame
            Data frame containing expressions data indexed by sample_id. Ideally it is output from get_df_from_table
            method on expressions tsv.

        mut_df: Pandas Data frame
            Data frame containing mutations data indexed by sample_id. Ideally it is output from get_df_from_table
            method on mutations tsv.

        Returns
        -------
        expr_df_fil: Pandas Data frame
            expressions data filtered by mutation
        mut_df_fil: Pandas Data frame
            mutation data filtered by specific mutation

        """

        print('Filtering to mutation {}'.format(mutation))
        # filters by individual mutation
        mut_df_fill = mut_df[mutation]

        # filters expressions df by sample_ids (index on both DFs) from mutations df
        expr_df_fill = expr_df.loc[mut_df_fill.index, :]

        return expr_df_fill, mut_df_fill

    def run(self, mutation):

        """

        Parameters
        ----------
        mutation: String
            String representation of mutation to be analyzed

        Returns
        -------
        expression_df_processed: Pandas Data frame
            Expressions data for one mutation, ready to be consumed by a machine-learning process
        mutation_df_processed: Pandas Data frame
            Mutation data for one mutation, ready to be consumed by a machine-learning process

        """

        expr_file = self._expressions_file
        mut_file = self._mutation_file
        self.download_files()
        expr_df_raw = self.get_df_from_table(expr_file)
        mut_df_raw = self.get_df_from_table(mut_file)
        expr_df_processed, mut_df_processed = self.filter_data_by_mutation(mutation, expr_df_raw, mut_df_raw)
        return expr_df_processed, mut_df_processed


