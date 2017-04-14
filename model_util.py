

def compile_model(model, X_train, y_train, epochs):
    print("Compiling the model..........................\n")
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=epochs)
    print("Saving the model..........................\n")
    model.save('model.h5')
