import json
from cognoml.analysis import CognomlClassifier
from cognoml.data import CognomlData

if __name__ == '__main__':
    a = CognomlData(mutations_json_url='https://github.com/cognoma/machine-learning/raw/876b8131bab46878cb49ae7243e459ec0acd2b47/data/api/hippo-input.json')
    x, y = a.run()
    classifier = CognomlClassifier(x, y)
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
