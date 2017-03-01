from unittest import mock

from cognoml import utils


@mock.patch('os.mkdir')
@mock.patch('os.path.exists')
def test_create_dir(mock_mkdir, mock_exists):
    mock_exists.return_value = False
    utils.create_dir('test_directory')
    mock_mkdir.assert_called_with('test_directory')
