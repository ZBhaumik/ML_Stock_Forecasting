#GRU-RNN Trading Agent
GRU_RNN_avg = (GRU_prediction[:len(GRU_prediction)-1]+rnn_predictions)/2
actual_pred_plot(GRU_RNN_avg)

capital = 0
predictions = GRU_RNN_avg[:,0]
saleprice = predictions[0]
stock = 10000/saleprice
for i in range(1, predictions.size):
  #print(str(capital) + " " + str(stock) + " " + str(saleprice) + " " + str(predictions[i]))
  if(predictions[i]>saleprice):
      prop_order = capital/saleprice
      stock += prop_order
      capital = 0
      saleprice = predictions[i]
  elif((predictions[i]<saleprice) and (stock!=0)):
        capital = stock*saleprice
        saleprice = predictions[i]
        stock = 0

capital += stock*predictions[len(predictions)-1]

print("Final Portfolio Value: " + str(capital))
print("% Return: " + str((-100*(1-capital/10000))))
print("Buy and Hold Return %: " + str(-100*(1-(predictions[len(predictions)-1]*(10000/predictions[0]))/10000)))