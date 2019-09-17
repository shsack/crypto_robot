# Bitcoin price forcast
Trading currencies is hard, especially Bitcoin which is one of the most volitile currencies in the world.
There are hundreds of technical indicators that usually give conflicting buy and sell signials.
Price seemingly goes and up and down randomly and it might seems like and impossible task to predict the price movement. 
This project aims at outsourcing the price forcast to a [recurrent neural network (RNN)](https://en.wikipedia.org/wiki/Recurrent_neural_network), 
more specifically a [long-short term memory (LSTM)](https://en.wikipedia.org/wiki/Long_short-term_memory) RNN.

## Training the LSTM
LSTMs lend themselves well for making predictions based on time series data, like the historical Bitcoin price.
The LSTM is implemented using [Keras](https://keras.io/) which provides a high level API to implement neural networks.
To train the network and plot the historical training history run `python train_evaluate_model.py`.

![loss_history](https://user-images.githubusercontent.com/45107198/65030673-37465000-d940-11e9-8deb-f575239156f3.png)

![performance_train_test](https://user-images.githubusercontent.com/45107198/65030726-4deca700-d940-11e9-8364-7dfdeb28f3d9.png)

![performance_test](https://user-images.githubusercontent.com/45107198/65030772-6492fe00-d940-11e9-9a13-f656e46d4e06.png)


## Backtest trading strategy
To further evaluate the quality of the forecast `backtest.py` implementes a simple trading strategy and tests it on the test dataset.
The trading strategy simply places a long or short position at the beginning of the day based on the forecast and closes the position at the end of the day.
No fees or spreads are taken into account. The preformance of the LSTM trading strategy is compared to simply buying and holding as well as random daily long and short positions. The results for backtesting on the 2019 bull market show promising results for the LSTM strategy. 

LSTM | Buy & Hold | Random
--- | --- | ---
261% | 252% | 99%
