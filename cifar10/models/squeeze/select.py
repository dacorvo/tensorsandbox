import tensorflow as tf

import squeeze0
import squeeze1

def by_name(name):
    if name == 'squeeze0':
        model = squeeze0.Squeeze0()
    if name == 'squeeze1':
        model = squeeze1.Squeeze1()
    else:
        raise ValueError('No such model %s' % name)
    return model
