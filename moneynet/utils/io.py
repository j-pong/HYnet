import os

def load_file_list(datadir):
    f = []
    for (dirpath, dirnames, filenames) in os.walk(datadir):
        for filename in filenames:
            f.append(os.path.join(dirpath, filename))

    return f