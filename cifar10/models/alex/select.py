import tensorflow as tf

import alex0
import alex1
import alex2

def by_name(name):
    if name == 'alex0':
        model = alex0.Alex0()
    elif name == 'alex1':
        model = alex1.Alex1()
    elif name == 'alex2':
        model = alex2.Alex2()
    else:
        raise ValueError('No such model %s' % name)
    return model
