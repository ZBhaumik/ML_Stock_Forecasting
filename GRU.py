def GRU_model(X_train, y_train, X_test, sc):
    from keras.models import Sequential
    from keras.layers import Dense, SimpleRNN, GRU
    from keras.optimizers import SGD
    my_GRU_model = Sequential()
    my_GRU_model.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
    my_GRU_model.add(GRU(units=50, activation='tanh'))
    my_GRU_model.add(Dense(units=2))
    my_GRU_model.compile(optimizer=SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=False),loss='mean_squared_error')
    my_GRU_model.fit(X_train,y_train,epochs=50,batch_size=150, verbose=0)
    GRU_prediction = my_GRU_model.predict(X_test)
    GRU_prediction = sc.inverse_transform(GRU_prediction)
    return my_GRU_model, GRU_prediction

X_train, y_train, X_test, sc = ts_train_test_normalize(all_data,5,1)
my_GRU_model, GRU_prediction = GRU_model(X_train, y_train, X_test, sc)
actual_pred_plot(GRU_prediction) 