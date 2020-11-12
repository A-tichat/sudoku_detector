import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

class Model(object):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.init_model()
    
    def init_model(self):
        model = Sequential()
        model.add(Conv2D(28, kernel_size=(3,3), input_shape=self.input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
        model.add(Dense(128, activation=tf.nn.relu))
        model.add(Dropout(0.2))
        model.add(Dense(10, activation=tf.nn.softmax))

        model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

        self.model = model

    def normalize_image(self, images):
        '''Normalizing the RGB codes by dividing it to the max RGB value.'''
        images = images.astype('float32')
        return images / 255

    def train(self, x_train, y_train, epochs):
        x_train = self.normalize_image(x_train)
        self.model.fit(x=x_train,y=y_train, epochs=epochs)

    def evaluate(self, x_test, y_test):
        x_test = self.normalize_image(x_test)
        return self.model.evaluate(x_test, y_test)

    def predict(self, image):
        image = self.normalize_image(image)
        return self.model.predict(image.reshape((1,) + self.input_shape))

    def load_model(self, path):
        self.model.load_weights(path)

    def save_model(self, path):
        self.model.save_weights(path)
