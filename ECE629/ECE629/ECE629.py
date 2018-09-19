import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

model = tf.keras.models.Sequential()

# We will create a single hidden layer
# And an output layer
# Since we want to classify the data the output layer will use a softmax

# Since mnist data  flattened is 28 x 28 = 784
# input layer must be 784 wide

# Add 64 hidden layer neurons
model.add(tf.keras.layers.Dense(64, activation='relu', input_dim=784))

# Output layer 
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# We need to compile the model 
# What does model.compile do?
# Ans: defines the loss function, the optimizer and the metrics
model.compile(
              optimizer = 'sgd',
              loss='mean_squared_error',
              metrics=['accuracy'])

# Now our model is built we need to train it
# But before that we need to make sure the data is in the right format
no_training_images = x_train.shape[0]
size_of_each_image = x_train.shape[1] * x_train.shape[2]

# Each image is 28 x 28 we will flatten in into a single 784 vector
x_train = x_train.reshape((no_training_images, size_of_each_image))

y_train = tf.keras.utils.to_categorical(y_train, 10)

# Now we can train out model
model.fit(x_train, y_train, epochs=50, verbose=1)

# Now we want to test out our neural net
# Each image is 28 x 28 we will flatten in into a single 784 vector
no_training_images = x_test.shape[0]
size_of_each_image = x_test.shape[1] * x_test.shape[2]
x_test = x_test.reshape((no_training_images, size_of_each_image))

y_test = tf.keras.utils.to_categorical(y_test, 10)
score = model.evaluate(x_test, y_test, batch_size=128)

print("Loss = " + str(score[0]) + " Accuracy = " + str(score[1]))