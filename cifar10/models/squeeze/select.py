import tensorflow as tf

import squeeze0

def by_name(name):
    if name == 'squeeze0':
        model = squeeze0.Squeeze0()
    else:
        raise ValueError('No such model %s' % name)
    return model
