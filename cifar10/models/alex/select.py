import tensorflow as tf

import alex0
import alex1

def by_name(name):
    if name == 'alex0':
        model = alex0.Alex0()
    elif name == 'alex1':
        model = alex1.Alex1()
    else:
        raise ValueError('No such model %s' % name)
    return model
