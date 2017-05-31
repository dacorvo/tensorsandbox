import tensorflow as tf

import nin0

def by_name(name):
    if name == 'nin0':
        model = nin0.Nin0()
    else:
        raise ValueError('No such model %s' % name)
    return model
