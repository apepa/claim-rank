from src.models.keras.keras_model import KerasModel
from keras.layers import *
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.optimizers import *
from keras.models import Model
from keras.initializers import *
from keras import regularizers

import time

class MultilayerModel(KerasModel):
  def __init__(self, prefix='model_multitask', epochs=100, focus=1,
            focus_value=3.5, lr=0.08, layers=1, size=1100, momentum=0.7, nesterov=True, batch_size=700,
            size_2=700, dropout_1=0.6, dropout_2=0, l2_1 = 0.002, l2_2=0.006, decay=1e-4
      ):
      KerasModel.__init__(self, prefix=prefix, 
          epochs=epochs,
          focus=focus,
          focus_value=focus_value,
          lr=lr,
          layers=layers,
          size=size,
          momentum=momentum,
          nesterov=nesterov,
          batch_size=batch_size,
          decay=decay)
      self.size_2 = size_2
      self.dropout_1 = dropout_1
      self.dropout_2 = dropout_2
      self.l2_1 = l2_1
      self.l2_2 = l2_2

  def create_model(self, in_count, out_count):
      loss_weights = dict(map(lambda i: ('pred_' + str(i), 1), range(out_count)))
      if (self.focus != -1):
          loss_weights['pred_' + str(self.focus)] = self.focus_value
      # shared layer
      input_layer = Input(shape=(in_count,), name='input')
      dense_shared = Dense(self.size, name='dense_shared', activation='linear',
                    kernel_initializer=lecun_normal(seed=42), kernel_regularizer=regularizers.l2(self.l2_1))(input_layer)
      lrelu_shared = LeakyReLU(name='shared_lrelu', alpha=0.1)(dense_shared)
      drop_shared = Dropout(self.dropout_1)(lrelu_shared)
      
      out_layers = []
      for i in range(out_count):
          # any layer
          dense = Dense(self.size_2, name='dense_' + str(i), activation='linear',
                         kernel_initializer=lecun_normal(seed=42), kernel_regularizer=regularizers.l2(self.l2_2))(drop_shared)
          lrelu = LeakyReLU(name='lrelu_' + str(i), alpha=0.1)(dense)
          drop = Dropout(self.dropout_2)(lrelu)
          pred = Dense(1, activation='sigmoid', name='pred_' + str(i), 
                           kernel_initializer=lecun_normal(seed=42), kernel_regularizer=regularizers.l2(self.l2_2))(drop)
          out_layers.append(pred)

      opt=SGD(nesterov=self.nesterov, momentum=self.momentum, lr=self.lr, decay=self.decay)
        
      model = Model(inputs=[input_layer], outputs=out_layers)
      model.compile(optimizer=opt, 
                    loss='binary_crossentropy',
                    loss_weights=loss_weights, # ! give more weight to PF's loss 
                    metrics=['accuracy'])
      return model

  def get_callbacks(self):
      # timestamp = int(time.time())
      # if (self.focus == -1):
      #   monitor_name = 'val_loss';
      # else:
      #   monitor_name = 'pred_' + str(self.focus) + '_loss';

      # weights_filename = self.prefix + str(timestamp)
      # # tensor_board_cb = TensorBoard(log_dir='./logs/'+run_name+str(timestamp))
      # model_checkpoint = ModelCheckpoint('./'+weights_filename, monitor=monitor_name, 
      #                                    save_best_only=True, verbose=0, save_weights_only=True)
      # early_stopping = EarlyStopping(monitor=monitor_name, patience=20, verbose=0)
      # return [model_checkpoint, early_stopping]
      return []