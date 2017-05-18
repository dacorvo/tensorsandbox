import tensorflow as tf

import alex0

def by_name(name):
    if name == 'alex0':
        model = alex0.Alex0()
    else:
        raise ValueError('No such model %s' % name)
    return model
