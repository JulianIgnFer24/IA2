import pickle


def load_data(path="processed_data.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)