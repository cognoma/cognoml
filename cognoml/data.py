import os
from urllib.request import urlretrieve
import json
import requests
import pandas as pd
from cognoml.utils import create_dir


class CognomlData:
    """Class to manage downloading, cleaning, and filtering data for the Cognoml Machine Learning project."""

    def __init__(self, article_id=3487685, figshare_url='https://api.figshare.com/v2/articles/{}/versions',
                 directory='download', covariates_file='covariates.tsv',
                 expressions_file='expression-matrix.tsv.bz2',
                 mutations_json_url = 'https://github.com/cognoma/machine-learning/raw/876b8131bab46878cb49ae7243e459ec0acd2b47/data/api/hippo-input.json',
                 version=None):
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
        self._covariates_file = covariates_file
        self._expressions_file = expressions_file
        self._files_to_download = [covariates_file, expressions_file]
        self._covariates_path = os.path.join(self._download_path, self._covariates_file)
        self._expressions_path = os.path.join(self._download_path, self._expressions_file)
        self._mutations_json_url = mutations_json_url
        self._mut_df = self.get_mutations_df()

    def get_mutations_df(self):
        """
        Get mutations df from front end rest API call

        Returns
        -------
        mut_df: Pandas Series with index 'sample_id' and column 'mutation_status'
        """
        url = self._mutations_json_url
        mut_df = pd.read_json(url).set_index('sample_id')
        processed_mut_df = mut_df['mutation_status']
        return processed_mut_df

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
        directory = self._directory

        # Create dirs
        create_dir(directory)
        create_dir(download_path)

        path = os.path.join(download_path, 'info.json')
        with open(path, 'w') as write_file:
            json.dump(article, write_file, indent=2, ensure_ascii=False, sort_keys=True)

        # Download the files specified by the class above
        for file_info in article['files']:
            name = file_info['name']
            if name in self._files_to_download:
                path = os.path.join(download_path, name)
                if os.path.exists(path):
                    print('{} already exists, checking next file'.format(path))
                    continue
                url = file_info['download_url']
                print('Downloading {} to `{}`'.format(url, name))
                urlretrieve(url, path)
            else:
                print('Not downloading {}, not needed at the moment'.format(name))

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
        pickle_dict = {file_name: file_name + '_pickle' for file_name in self._files_to_download}
        try:
            pickle_file = pickle_dict[tsv_file]
        except KeyError:
            raise KeyError('Not expected input, choose either {0} or {1}'.format(self._covariates_file,
                                                                                 self._expressions_file))
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
    def filter_data_by_mutation(expr_df, mut_df):
        """
        Filters total data sets to include only data for a single mutation

        Parameters
        ----------
        expr_df: Pandas Data frame
            Data frame containing expressions data indexed by sample_id. Ideally it is output from get_df_from_table
            method on expressions tsv.

        mut_df: Pandas Data frame
            Data frame containing mutations data indexed by sample_id. This is read in from the front end in the form
            of a json that is converted to a df.

        Returns
        -------
        expr_df_fil: Pandas Data frame
            expressions data filtered by mutations selected by front end

        """

        # filters expressions df by sample_ids (index on both DFs) from mutations df
        expr_df_fil = expr_df.loc[mut_df.index, :]
        return expr_df_fil

    def run(self):

        """
        Returns
        -------
        expression_df_processed: Pandas Data frame
            Expressions data for one mutation, ready to be consumed by a machine-learning process
        mutation_df_processed: Pandas Data frame
            Mutation data for one mutation, ready to be consumed by a machine-learning process

        """

        expr_file = self._expressions_file
        self.download_files()
        expr_df_raw = self.get_df_from_table(expr_file)
        mut_df = self._mut_df
        expr_df_processed = self.filter_data_by_mutation(expr_df_raw, mut_df)
        return expr_df_processed, mut_df


