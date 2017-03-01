import re

from cognoml import data


def test_get_article_versions():
    url = r'https://api.figshare.com/v2/articles/([0-9]*)/versions/([0-9]*$)'
    url_pattern = re.compile(url, re.MULTILINE)
    cognoml_data = data.CognomlData(
        article_id=3487685,
        expressions_file='expression-matrix.tsv.bz2',
        figshare_url='https://api.figshare.com/v2/articles/{}/versions',
        version=None,
        directory='download',
        mutations_json_url=('https://github.com/cognoma/machine-learning/raw/'
                            '876b8131bab46878cb49ae7243e459ec0acd2b47/data/'
                            'api/hippo-input.json'),
        covariates_file='covariates.tsv')
    version_to_url = cognoml_data.get_article_version()
    assert type(version_to_url) == dict
    for key, value in version_to_url.items():
        assert url_pattern.match(value) is not None
