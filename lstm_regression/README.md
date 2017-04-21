# LSTM for regression problems with adjustable lookback

This script have been adopted from the excellent tutorial provided [here](http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/)

### HOW TO USE

    import lstm_reg
    
    lstm_reg.run('data')

### INPUT
a series of time-series integers or floats

### OUTPUT
training result plot and RMSE

### COMMENTS 
I've tried this with different lookbacks and it works great but it's better to run at least 20 epoches with small data. Excellent results with even just 150 samples 
