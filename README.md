# Bitcoin-Private-Key-Predictor-Tranformer

Transformer based AI Bitcoin Private key predictor

This code is a poof of concept !!

It tries to predict the bitcoin private key for a inputed address.

the file "gen_data" will create the dataset.csv containing bitcoin addresses and the corresponding private keys.

dataset content example:

target,input

67b30346dfc48f7ea7dca731ea0a9c6d772698bc79c0f1f70c2df0ce04db6ec7,16xWWJAJfX4M2xFkr5VqTyAN8buMTTg8qd
98f79ae2e0e01c60d8cd1b276e4c6d28c1f7eea982d129a89502f799d7c8d879,1L56PZZwXFfFW7woKroSssx7Rba7sLMxXT
76b5f918ecd7f167484632cc0a3532ad3a21e2904a7c35d72891b71e75846f81,1DEhz2Fct4MSku7GDUZ93W8aoFhCff8Pjv

...


the file "train" will create and train the Transformer model in the dataset.csv.

the file "pred" will load the trained model and allow the user to input a bitcoin address and will output its predictment to be the coresponding private key.

for more large the dataset be more accurate the model can become.

If ur pc have an avaiable GPU it will use her , if not it will use cpu.
