import os, time
import tensorflow as tf


print("TF:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))




gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("memory_growth=ON")
    except Exception as e:
        print("memory_growth set failed:", e)




N = 4096
with tf.device("/GPU:0"):
    a = tf.random.normal((N, N))
    b = tf.random.normal((N, N))
    c = tf.matmul(a, b)     # warmup
    _ = c.numpy()           


t0 = time.time()
with tf.device("/GPU:0"):
    for _ in range(8):
        c = tf.matmul(a, b)
    _ = c.numpy()


print("elapsed_sec:", time.time() - t0)
print("done.")
print(tf.config.get_visible_devices("GPU"))
