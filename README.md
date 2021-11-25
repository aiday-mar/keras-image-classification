# Keras Image Classification

This is a Jupyter notebook containing code written in Python, using the Keras library, to do image classification of a set of numbers. To do this, we built (convolutional) artificial neural networks which we trained before testing. The link to the Google collab notebook is as follows: https://colab.research.google.com/drive/1nvQR74HlDLEznBvyyxySYHBKcd9CVSjD?usp=sharing.

During my masters degree, I have taken the course Artificial Neural Networks. The project of the course was to classify first the MNIST dataset of numbers, then the fashion-MNIST dataset, using (convolutional) neural networks. To this end, we use the Keras and Tensorflow libraries of Python. We changed the various parameters of the neural networks to find out which set of parameters gave us the lowest testing error. The parameters studied were: number of hidden layers, type of optimizer (SGD or Adam), size of the learning rate, number of hidden neurons per layer, type of regularization, size of the regularization constant. Below is the code for the final convolutional neural networks used in the last question of the notebook. First we load the data and reshape it:

```
(x_fashion_train_conv, y_fashion_train_conv), (x_fashion_test_conv, y_fashion_test_conv) = keras.datasets.fashion_mnist.load_data()
x_fashion_train_conv = np.expand_dims(x_fashion_train_conv, -1)
x_fashion_test_conv = np.expand_dims(x_fashion_test_conv, -1)
y_fashion_train_conv = y_fashion_train
y_fashion_test_conv = y_fashion_test
```

Then we provide the method to build the network

```
def build_convolutional_network(number_of_hidden_layers = 0, units = 100, optimizer="sgd", lr = 0.01, momentum = 0.9, dropOut = None, regularization_kernel = 0, regularization_bias = 0, batchNormalization = False) :
    model = Sequential()
    model.add(Conv2D(units, (3, 3), activation="relu", input_shape=(28, 28, 1), kernel_regularizer=regularizers.l2(regularization_kernel), bias_regularizer=regularizers.l2(regularization_bias)))
    model.add(MaxPooling2D((2, 2)))
    
    if(dropOut != None):
        model.add(Dropout(dropOut))
    if(batchNormalization == True):
        model.add(BatchNormalization())

    if(number_of_hidden_layers > 0) :
        for i in range(number_of_hidden_layers):
            model.add(Conv2D(units, (3, 3), activation="relu", input_shape=(28, 28, 1), kernel_regularizer=regularizers.l2(regularization_kernel), bias_regularizer=regularizers.l2(regularization_bias)))
            model.add(MaxPooling2D((2, 2)))

            if(dropOut != None):
                model.add(Dropout(dropOut))
            if(batchNormalization == True):
                model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(10, activation="softmax"))

    opt = keras.optimizers.SGD(lr=lr, momentum=momentum)
    if(optimizer == "adam"):
        opt = keras.optimizers.Adam(learning_rate = lr)

    model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=opt)
    return model
```

Then we build three different neural networks and we compare them

```
model_batch = build_convolutional_network(number_of_hidden_layers = 1, units = 100, optimizer="adam", lr = 0.01, batchNormalization=True)
history_batch = model_batch.fit(x_fashion_train_conv, y_fashion_train_conv, batch_size=128, epochs=20, verbose=0, validation_data=(x_fashion_test_conv, y_fashion_test_conv))
print("Best validation accuracy for a convolutional neural network with batch normalization: " + str(max(history_batch.history["val_accuracy"])))

model_drop = build_convolutional_network(number_of_hidden_layers = 1, units = 100, optimizer="adam", lr = 0.01, dropOut=0.1)
history_drop = model_drop.fit(x_fashion_train_conv, y_fashion_train_conv, batch_size=128, epochs=20, verbose=0, validation_data=(x_fashion_test_conv, y_fashion_test_conv))
print("Best validation accuracy for a convolutional neural network with dropout of 0.1 : " + str(max(history_drop.history["val_accuracy"])))

model_reg = build_convolutional_network(number_of_hidden_layers = 1, units = 100, optimizer="adam", lr = 0.01, regularization_kernel = 0.01, regularization_bias = 0.01)
history_reg = model_reg.fit(x_fashion_train_conv, y_fashion_train_conv, batch_size=128, epochs=20, verbose=0, validation_data=(x_fashion_test_conv, y_fashion_test_conv))
print("Best validation accuracy for a convolutional neural network with regularization on weights and bias of 0.01: " + str(max(history_reg.history["val_accuracy"])))
plot_history(history_batch,"With batch normalization")
plot_history(history_drop,"With dropout of 0.1")
plot_history(history_reg,"With kernel and bias regularization of 0.01")
```

The comparative plots of the three different neural networks are as follows. We can see a comparison of the validation and testing accuracies and cross-entropies.

![alt text](https://github.com/aiday-mar/Keras-Image-Classification/blob/main/nn1.PNG?raw=true)

![alt text](https://github.com/aiday-mar/Keras-Image-Classification/blob/main/nn2.PNG?raw=true)

![alt text](https://github.com/aiday-mar/Keras-Image-Classification/blob/main/nn3.PNG?raw=true)

We then construct an optimal convolutional neural network and compare it with a basic neural network.

```
model_optimal = build_convolutional_network(number_of_hidden_layers = 1, units = 100, optimizer="adam", batchNormalization=True, dropOut=0.1)
history_optimal = model_optimal.fit(x_fashion_train_conv, y_fashion_train_conv, batch_size=128, epochs=20, verbose=0, validation_data=(x_fashion_test_conv, y_fashion_test_conv))

fig_comparison = comparison_plot(history_6_1, history_optimal, "Naive model", "Optimized model", "Effect of tricks and regularization")
fig_comparison.set_size_inches(20,8)
```
The following plots are obtained.

![alt text](https://github.com/aiday-mar/Keras-Image-Classification/blob/main/nn4.PNG?raw=true)
