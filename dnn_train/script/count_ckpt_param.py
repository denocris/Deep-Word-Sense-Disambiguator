#!/usr/bin/env python

import sys
import tensorflow as tf
import numpy as np

if len(sys.argv) == 2:
    ckpt_fpath = sys.argv[1]
else:
    print('Usage: python count_ckpt_param.py trained_models/my-model/model.ckpt-24676')
    sys.exit(1)


reader = tf.train.NewCheckpointReader(ckpt_fpath)
# Open TensorFlow ckpt
#reader = tf.train.NewCheckpointReader('trained_models/cnn16lstm8-gloff-emb20-bs512-lr5em2drop1em4rnd_135epochs_voc20k_v3/model.ckpt-24676')
#reader = tf.train.NewCheckpointReader('trained_models/cnn16lstm8-gloff-emb20-bs512-lr5em2drop1em4rnd_135epochs_voc20k_v3/model.ckpt-24676')

print('\nCount the number of parameters in ckpt file(%s)' % ckpt_fpath)
param_map = reader.get_variable_to_shape_map()
total_count = 0
for k, v in param_map.items():
    if 'Momentum' not in k and 'global_step' not in k:
        temp = np.prod(v)
        total_count += temp
        print('%s: %s => %d' % (k, str(v), temp))

print('Total Param Count: %d' % total_count)
