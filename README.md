# Bitcoin price forcast
Trading currencies is hard, especially Bitcoin which is one of the most volitile currencies in the world.
There are hundreds of technical indicators that usually give conflicting buy and sell signials.
Price seemingly goes and up and down randomly and it might seems like and impossible task to predict the price movement. 
This project aims at outsourcing the price forcast to a [recurrent neural network (RNN)](https://en.wikipedia.org/wiki/Recurrent_neural_network), 
more specifically a [long-short term memory (LSTM)](https://en.wikipedia.org/wiki/Long_short-term_memory) RNN.

## Training the LSTM
LSTMs lend themselves well for making predictions based on time series data, like the historical Bitcoin price.
The LSTM is implemented using [Keras](https://keras.io/) which provides a high level API to implement neural networks.
To train and evaluate the network on historical data run `python train_evaluate_model.py`. 

## Backtest trading strategy
To further evaluate the quality of the forecast `backtest.py` implementes a simple trading strategy and tests it on the test dataset.
The trading strategy simply places a long or short position at the beginning of the day based on the forecast and closes the position at the end of the day.
No fees or spreads are taken into account in the backtest. The preformance of the LSTM trading strategy is compared to simply buying and holding
as well as random daily long and short positions. 


