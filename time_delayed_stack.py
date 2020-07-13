import tensorflow as tf
import tensorflow.keras as K


class TimeStack(tf.keras.layers.Layer):
  '''Time stack layer'''

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.rnn_state_size = 16
    self.lstm_t = K.layers.LSTM(self.rnn_state_size, return_sequences=True) # rnn running along time 
    self.lstm_ffb = K.layers.Bidirectional(
                    K.layers.LSTM(self.rnn_state_size)) # bi-rnn running along freq. (forward and backward)

  def call(self, x):
    B, T, F, D_in = x.shape  
    # Implement the time-stack here...
    # same pre-padding 
    x_slice = K.layers.Lambda(lambda x: x[:, 0, :, :])(x)
    x_slice = K.backend.expand_dims(x_slice, axis=1)
    x_padded = K.backend.concatenate((x_slice, x), axis=1)  # B x (T+1) x F x D_in

    x_t = K.backend.permute_dimensions(x_padded, (0, 2, 1, 3)) # B x F x (T+1) x D_in
    x_t = K.backend.reshape(x_t, (B*F, T+1, D_in)) 
    y1 = self.lstm_t(x_t)
    y1 = K.backend.reshape(y1, (B, F, T+1, self.rnn_state_size))
    y1 = K.backend.permute_dimensions(y1, (0, 2, 1, 3)) # B x (T+1) x F x H
    y1 = K.backend.sum(y1, axis=2) # B x (T+1) x H
    y1 = K.backend.expand_dims(y1, axis=2) # B x (T+1) x 1 x H
    y1 = K.backend.repeat_elements(y1, F, axis=2) # B x (T+1) x F x H
    y1 = K.layers.Lambda(lambda x: x[:, 1:, :, :])(y1) # B x T x F x H
    print(y1.shape)

    x_f = K.backend.reshape(x_padded, (B*(T+1), F, D_in))
    y2 = self.lstm_ffb(x_f)
    y2 = K.backend.reshape(y2, (B, T+1, 2*self.rnn_state_size)) # B x (T+1) x 2*H
    y2 = K.backend.cumsum(y2, axis=1)
    y2 = K.backend.expand_dims(y2, axis=2) # B x (T+1) x 1 x 2*H
    y2 = K.backend.repeat_elements(y2, F, axis=2) # B x (T+1) x F x 2*H
    y2 = K.layers.Lambda(lambda x: x[:, 1:, :, :])(y2) # B x T x F x 2*H
    print(y2.shape)
    
    y = K.backend.concatenate((y1, y2), axis=-1) # B x T x F x 3*H
    return y
  

if __name__ == "__main__":
  import numpy as np 
  tstack_layer = TimeStack()
  x = tf.random.normal([8, 10, 12, 20])
  print('input shape: {}'.format(x.shape))
  out = tstack_layer(x)
  print('output shape: {}'.format(out.shape))
