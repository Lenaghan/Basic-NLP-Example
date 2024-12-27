import time
import pickle
from bagofwords import OHE_BOW

def bow_encode_with_cv(cleaned_train, cleaned_test, train_filename, test_filename):
    """
    Fit CountVectorizer to training data and encode both training and test data.
    Args:
        cleaned_train: list of N strings (training dataset)
        cleaned_test: list of M strings (test dataset)
        train_filename: filename to save transformed training data
        test_filename: filename to save transformed test data
    """
    ohe_bow = OHE_BOW()
    start_time = time.time() # start the timer

    # Fit the vectorizer on the training data and transform the training and test data
    train_encoded = ohe_bow.cv_bow_transform(cleaned_train, fit=True)
    test_encoded = ohe_bow.cv_bow_transform(cleaned_test, fit=False)
    compute_time = time.time() - start_time

    # Save the transformed data to files
    with open(f'./data/{train_filename}', 'wb') as file:
        pickle.dump(train_encoded, file)
    with open(f'./data/{test_filename}', 'wb') as file:
        pickle.dump(test_encoded, file)

    # Print dataset shape and time taken
    print(f'CountVectorizer Bag of Words representation shape (train): {train_encoded.shape}')
    print(f'CountVectorizer Bag of Words representation shape (test): {test_encoded.shape}')
    print(f"Time to encode dataset: {compute_time:.2f} seconds\n")
