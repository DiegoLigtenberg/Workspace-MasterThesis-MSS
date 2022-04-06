import tensorflow as tf
build = tf.sysconfig.get_build_info()
print(build["cuda_version"])
print(build["cudnn_version"])

print("num gpu available: ",len(tf.config.experimental.list_physical_devices("GPU")))