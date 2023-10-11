import json
from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train):
    # Load the configuration from the JSON file
    with open('config.json', 'r', encoding='utf-8') as file:
        config = json.load(file)

    # Check the classifier specified in the configuration
    if config['classifier'] == 'random_forest':
        classifier = RandomForestClassifier(**config['random_forest'])
    else:
        raise ValueError(f"Unknown classifier: {config['classifier']}")

    # Train the classifier
    classifier.fit(X_train, y_train)
    return classifier
