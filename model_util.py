import matplotlib.pyplot as plt


def compile_model(model, X_train, y_train, epochs):
    print("Compiling the model..........................\n")
    model.compile(loss='mse', optimizer='adam')
    history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=epochs)
    plot_model_history(history_object)
    print("Saving the model..........................\n")
    model.save('model.h5')


def compile_model_generator(model, train_samples, validation_samples, train_generator, validation_generator, epochs):
    print("Compiling the model..........................\n")
    model.compile(loss='mse', optimizer='adam')
    history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=epochs)
    plot_model_history(history_object)
    print("Saving the model..........................\n")
    model.save('model.h5')
    # Summarizing the model
    model.summary()


def plot_model_history(history_object):
    # print the keys contained in the history object
    print(history_object.history.keys())

    # plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
    return

# #model.fit_generator(train_generator, samples_per_epoch =
#     len(train_samples), validation_data =
#     validation_generator,
#     nb_val_samples = len(validation_samples),
#     nb_epoch=5, verbose=1)