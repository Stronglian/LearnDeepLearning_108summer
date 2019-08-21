# -*- coding: utf-8 -*-

from keras.models import load_model, save_weights, load_weights

# save
print('test before save: ', model.predict(X_test[0:2]))
model.save('my_model.h5')   # HDF5 file, you have to pip3 install h5py if don't have it
del model  # deletes the existing model

"""
test before save:  [[ 1.87243938] [ 2.20500779]]
"""

# load
model = load_model('my_model.h5')
print('test after load: ', model.predict(X_test[0:2]))

# save and load weights
model.save_weights('my_model_weights.h5')
model.load_weights('my_model_weights.h5')
