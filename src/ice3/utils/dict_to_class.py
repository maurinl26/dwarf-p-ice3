class DictToClass(object):
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

