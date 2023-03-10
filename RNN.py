def simple_rnn_model(X_train, y_train, X_test, sc):
    from keras.models import Sequential
    from keras.layers import Dense, SimpleRNN
    my_rnn_model = Sequential()
    my_rnn_model.add(SimpleRNN(32, return_sequences=True))
    my_rnn_model.add(SimpleRNN(64, return_sequences=True))
    my_rnn_model.add(SimpleRNN(32))
    my_rnn_model.add(Dense(2)) # The time step of the output
    my_rnn_model.compile(optimizer='rmsprop', loss='mean_squared_error')
    my_rnn_model.fit(X_train, y_train, epochs=200, batch_size=150, verbose=0)
    rnn_predictions = my_rnn_model.predict(X_test)
    from sklearn.preprocessing import MinMaxScaler
    rnn_predictions = sc.inverse_transform(rnn_predictions)
    return my_rnn_model, rnn_predictions

X_train, y_train, X_test, sc = ts_train_test_normalize(all_data,5,2)
my_rnn_model, rnn_predictions = simple_rnn_model(X_train, y_train, X_test, sc)
actual_pred_plot(rnn_predictions)