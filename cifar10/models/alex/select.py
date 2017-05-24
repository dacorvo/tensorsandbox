import tensorflow as tf

import alex0
import alex1
import alex2
import alex3
import alex4

def by_name(name):
    if name == 'alex0':
        model = alex0.Alex0()
    elif name == 'alex1':
        model = alex1.Alex1()
    elif name == 'alex2':
        model = alex2.Alex2()
    elif name == 'alex3':
        model = alex3.Alex3()
    elif name == 'alex4':
        model = alex4.Alex4()
    else:
        raise ValueError('No such model %s' % name)
    return model
