import tensorflow as tf
import matplotlib.pyplot as plt
import config

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=config.warmup_steps):
    super(CustomSchedule, self).__init__()
  
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)
    self.warmup_steps = warmup_steps
    
  def __call__(self, step):
    print(step)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

# Plot an example
def plot_learning_rate():
  temp_learning_rate_schedule = CustomSchedule(config.d_model)
  plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
  plt.ylabel("Learning Rate")
  plt.xlabel("Train Step")