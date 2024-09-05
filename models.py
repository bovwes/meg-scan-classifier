import tensorflow as tf
from tensorflow.keras import regularizers

class RNNModel:
    def __init__(self, input_shape, units_1=32, units_2=32, num_classes=4, dropout_rate=0.5, l2_reg=0.001):
        self.input_shape = input_shape
        self.units_1 = units_1
        self.units_2 = units_2
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg

    def build_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.SimpleRNN(units=self.units_1, activation='tanh',
                                            input_shape=self.input_shape, return_sequences=True,
                                            kernel_regularizer=regularizers.l2(self.l2_reg)))
        #model.add(tf.keras.layers.Dropout(self.dropout_rate))
        model.add(tf.keras.layers.SimpleRNN(units=self.units_2, activation='tanh',
                                            kernel_regularizer=regularizers.l2(self.l2_reg)))
        #model.add(tf.keras.layers.Dropout(self.dropout_rate))
        model.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))

        return model
