import json
from cognoml.analysis import CognomlClassifier
from cognoml.data import CognomlData
import logging

__CONFIG_FILE__ = 'default_config.json'


def main():
    """
    Run entire Cognoml process using default data
    """
    logger = logging.getLogger('cognoml')
    logger.setLevel(logging.INFO)

    # create the logging file handler
    fh = logging.FileHandler('cognoml.log')

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s ' +
                                  '- %(message)s')
    fh.setFormatter(formatter)

    # add handler to logger object
    logger.addHandler(fh)

    logger.info('Program started')

    with open(__CONFIG_FILE__, 'r') as f:
        config = json.load(f)

    a = CognomlData(**config)
    x, y = a.run()
    classifier = CognomlClassifier(x, y)
    classifier.fit()
    results = classifier.get_results()
    json_results = json.dumps(results, indent=2)
    print(json_results)

    logger.info('Cognoml has finished')


if __name__ == '__main__':
    main()
