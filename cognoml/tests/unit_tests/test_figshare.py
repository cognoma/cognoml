import os
import re
from cognoml import figshare

def test_get_article_versions():
    article_url = 'https://api.figshare.com/v2/articles/3487685/versions'
    url_pattern = re.compile("https://api.figshare.com/v2/articles/([0-9]*)/versions/([0-9]*$)", re.MULTILINE)
    version_to_url = figshare.get_article_versions(3487685)
    assert type(version_to_url) == dict
    for key, value in version_to_url.items():
        assert url_pattern.match(value) is not None

