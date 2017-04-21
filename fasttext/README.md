# Deep Learning / Neural Network for making integer value predictions from unstructured data / text


### Depencies 

Once you have Autonomio running, this will run too. Also you need to have the function files in the same directory where autonomio module folder is located. 

### How does it work? 

You provide a dataframe with one column text and one column the indepdent variable. Unlike Autonomio, this model yields good results for non categorical predictions as well. 

     from fasttext import FastText
     
     FastText(data[['text_col','var_col']],'var_col')

### Original Source 

Heavily modified from [Keras Examples Page](https://github.com/fchollet/keras/blob/master/examples/imdb_fasttext.py) and then restructured in to a class. 
