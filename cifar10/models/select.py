import tensorflow as tf

import cs231n

tf.app.flags.DEFINE_string('model', 'cnn',
                               """One of [cs231n, cnn, tuto].""")

def by_name(name):
    if name == 'cs231n':
        model = cs231n.Cs231n()
    else:
        raise ValueError('No such model %s' % name)
    return model
