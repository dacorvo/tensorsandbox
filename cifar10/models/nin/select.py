import tensorflow as tf

import nin0
import nin1
import nin2

def by_name(name):
    if name == 'nin0':
        model = nin0.Nin0()
    elif name == 'nin1':
        model = nin1.Nin1()
    elif name == 'nin2':
        model = nin2.Nin2()
    else:
        raise ValueError('No such model %s' % name)
    return model
