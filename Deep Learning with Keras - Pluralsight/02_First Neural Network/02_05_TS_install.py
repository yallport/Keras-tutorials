# To verify that Tensorflow is working
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Print version
print("Tensorflow version is: " + str(tf.__version__))

# Verify session works
hello = tf.constant('Hello from TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
