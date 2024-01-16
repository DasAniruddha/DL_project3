## Spam filter for Quora questions
### This is project no. 3
 In this project the task is to filter the questions asked in quora to see if they are spam or not.

 The projcct is a binary classification as such:
 * spam:0
 * not spam:1

 Before we applying any deep learning algorith to it we first need to process each question to clean and remove some of the following things as such:
 * convert the entire sentences to lower case
 * remove any punctuation
 * white space removal
 * remove any new line
 * remove number
 etc,

 Now the process is to create features out of word, we accomplished it using (glove embedding)[http://nlp.stanford.edu/data/glove.42B.300d.zip].

 ```
 embeding_index={}

 f=open('glove.42B.300d.txt',encoding='utf-8')

 for line in f:
    values=line.split()
    word=values[0]
    coefs=np.asarray(values[1:],dtype='float32')
    embeding_index[word]=coefs
 f.close()
 ```

 And using the same embedding we are going to create weights for our network:
 ```
 embedding_matrix=np.zeros((vocab_size+1,300))

 for word,i in tk.word_index.items():
    embed_vector=embeding_index.get(word)
    if embed_vector is not None:
        embedding_matrix[i]=embed_vector
 ```

 Finally we will use these in the network as such:
 ```
 inputs=Input(name='text_input',shape=[max_len,])

 embed=Embedding(vocab_size+1,
                300,
                input_length=max_len,
                mask_zero=True,
                weights=[embedding_matrix],
                trainable=False)(inputs)

 x = Bidirectional(GRU(64, return_sequences=True))(embed)
 x = GlobalMaxPool1D()(x)
 x = Dense(16, activation="relu")(x)
 x = Dropout(0.1)(x)
 final_layer = Dense(1, activation="sigmoid")(x)

 model=Model(inputs=inputs,outputs=final_layer)
 ```