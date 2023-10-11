from data.fetch_data import fetch_pre_purchase_data, fetch_post_trip_data
from data.preprocess_data import preprocess_data
from features.scaling_encoding import scale_and_encode
from models.train_model import train_model
from models.predict_model import make_predictions
from sklearn.metrics import classification_report
import pickle
from utils.config_loader import load_config

def main():
    # Load configuration (if you're using a config loader)
    config = load_config()
    
    # Fetch data
    pre_df = fetch_pre_purchase_data()
    post_df = fetch_post_trip_data()

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(pre_df, post_df)

    # Scaling and encoding
    X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded = scale_and_encode(X_train, X_test, y_train, y_test)

    # Train model
    classifier = train_model(X_train_scaled, y_train_encoded)

    # Make predictions
    y_pred = make_predictions(classifier, X_test_scaled)

    # Evaluate the model
    eval_report = classification_report(y_test_encoded, y_pred)
    
        # Save the evaluation report to a text file
    with open("model_evaluation.txt", "w") as f:
        f.write(eval_report)

    # Save the model
    with open("model.pkl", "wb") as f:
        pickle.dump(classifier, f)
        
if __name__ == "__main__":
    main()
