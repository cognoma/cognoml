import re

from cognoml import data


def test_get_article_versions():
    url = r'https://api.figshare.com/v2/articles/([0-9]*)/versions/([0-9]*$)'
    url_pattern = re.compile(url, re.MULTILINE)
    cognoml_data = data.CognomlData()
    version_to_url = cognoml_data.get_article_version()
    assert type(version_to_url) == dict
    for key, value in version_to_url.items():
        assert url_pattern.match(value) is not None
