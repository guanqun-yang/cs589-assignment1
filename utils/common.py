import pickle
import pathlib


def load_text_file(filename):
    if not pathlib.Path(filename).exists(): 
        print("FILE NOT EXISTENT")
        return None

    with open(filename, "r", errors="ignore") as fp:
        return fp.readlines()


def save_pickle_file(data, filename):
    with open(filename, "wb") as fp:
        pickle.dump(data, fp)


def load_pickle_file(filename):
    data = None
    with open(filename, "rb") as fp:
        data = pickle.load(fp)

    return data