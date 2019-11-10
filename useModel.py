import tensorflow as tf
from tensorflow import keras


new_model = keras.models.load_model('final.42-0.0398.hdf5')
new_model.summary()

# loss, acc = new_model.evaluate(test_images, test_labels)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))