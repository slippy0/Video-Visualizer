from dummy_loader import DummyLoader
from charades_loader import CharadesLoader

data_loaders = {'dummy': DummyLoader, 'charades': CharadesLoader}

def get_data_loader(loader_name):
    if loader_name not in data_loaders:
        raise KeyError('Invalid loader {}. Must be one of {}'.format(
            loader_name, data_loaders.keys()))
    return data_loaders[loader_name]
