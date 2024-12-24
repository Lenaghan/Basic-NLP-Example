import time
import pickle
from bagofwords import OHE_BOW

def bow_encode_with_cv(cleaned_train, cleaned_test, train_filename, test_filename):
    ohe_bow = OHE_BOW()
    start_time = time.time()
    train_encoded = ohe_bow.cv_bow_transform(cleaned_train, fit=True)
    test_encoded = ohe_bow.cv_bow_transform(cleaned_test, fit=False)
    compute_time = time.time() - start_time

    with open(train_filename, 'wb') as file:
        pickle.dump(train_encoded, file)
    with open(test_filename, 'wb') as file:
        pickle.dump(test_encoded, file)

    print(f'CountVectorizer Bag of Words representation shape (train): {train_encoded.shape}')
    print(f'CountVectorizer Bag of Words representation shape (test): {test_encoded.shape}')
    print(f"Time to encode dataset: {compute_time:.2f} seconds\n")
