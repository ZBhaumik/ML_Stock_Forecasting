def simple_rnn_model(X_train, y_train, X_test, sc):
    from keras.models import Sequential
    from keras.layers import Dense, SimpleRNN
    my_rnn_model = Sequential()
    my_rnn_model.add(SimpleRNN(32, return_sequences=True))
    my_rnn_model.add(SimpleRNN(32))
    my_rnn_model.add(Dense(2)) # The time step of the output
    my_rnn_model.compile(optimizer='rmsprop', loss='mean_squared_error')
    my_rnn_model.fit(X_train, y_train, epochs=100, batch_size=150, verbose=0)
    rnn_predictions = my_rnn_model.predict(X_test)
    from sklearn.preprocessing import MinMaxScaler
    rnn_predictions = sc.inverse_transform(rnn_predictions)
    return my_rnn_model, rnn_predictions

my_rnn_model, rnn_predictions_2 = simple_rnn_model(X_train, y_train, X_test, sc)
rnn_predictions_2[1:10]
actual_pred_plot(rnn_predictions_2)