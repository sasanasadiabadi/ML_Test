import tensorflow as tf
import tensorflow.keras as K


class TimeStack(tf.keras.layers.Layer):
  '''Time stack layer'''

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.rnn_state_size = 16
    self.lstm_t = K.layers.LSTM(self.rnn_state_size, return_sequences=True) # rnn running along time 
    self.lstm_ffb = K.layers.Bidirectional(
                    K.layers.LSTM(self.rnn_state_size, return_sequences=True)) # bi-rnn running along freq. (forward and backward)


  def call(self, x):
    B, T, F, D_in = x.shape
    # time-delayed stack 
    x_t = K.backend.permute_dimensions(x, (0, 2, 1, 3)) # B x F x T x D_in
    x_t = K.backend.reshape(x_t, (B*F, T, D_in))
    y1 = self.lstm_t(x_t)
    y1 = K.backend.reshape(y1, (B, F, T, self.rnn_state_size))
    y1 = K.backend.permute_dimensions(y1, (0, 2, 1, 3)) # B x T x F x H
    print(y1.shape)

    x_f = K.backend.reshape(x, (B*T, F, D_in))
    y2 = self.lstm_ffb(x_f)
    y2 = K.backend.reshape(y2, (B, T, F, 2*self.rnn_state_size)) # B x T x F x 2*H
    print(y2.shape)
    
    y = K.backend.concatenate((y1, y2), axis=-1) # B x T x F x 3*H

    return y
  


if __name__ == "__main__":
  tstack_layer = TimeStack()
  x = tf.random.normal([8, 10, 12, 20])
  out = tstack_layer(x)
  print(out.shape)
