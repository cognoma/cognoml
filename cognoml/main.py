import json

import requests

from cognoml.analysis import CognomlClassifier
from cognoml.data import CognomlData

if __name__ == '__main__':
    data = CognomlData()
    x, y = data.run('1')
    y_test = y.head(5000)
    x_test = pd.DataFrame(x[x.index.isin(list(y_test.index))])
    classifier = CognomlClassifier(x_test, y_test)
    classifier.fit()
    results = classifier.get_results()
    json_results = json.dumps(results, indent=2)
    print(json_results)
    # Create a classifier using mock input. Print output to stdout.
    #url = 'https://github.com/cognoma/machine-learning/raw/876b8131bab46878cb49ae7243e459ec0acd2b47/data/api/hippo-input.json'
    #response = requests.get(url)
    #payload = response.json()
    #payload['data_version'] = 4
    #results = classify(**payload, json_sanitize=True)
    #json_results = json.dumps(results, indent=2)
    #print(json_results)
