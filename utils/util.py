import os


def make_dirs(paths):
    assert isinstance(paths, list) or isinstance(paths, str), 'The paths should be string or list of string'
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
