# how to handle categorical data for a neural network
neural network has a hard time working with categorical data and for data with man options this only makes it harder as for counterye, there can be 41 different dummy variables. there are many solutions to handlethis problem lets look at some of them

the data should be standarized with any neural network not only deep learning.

## ordinary encoding
have a numbr for each countery Norway=1, Sweeden=2, Engladn=4 ect.  this are a weird way to encode the data as Norway != 1/2 * sweeden. but it will make the newural network run and probably not have soo god results.

## One-hot encoding
the idea here is that we have a vector of possible value and each categori has a index in the vector as 1 and thre rest 0 e.i Norway=[1,0,0], sweeden=[0,1,0] and england=[0,0,1]. with this we can amke a matrix with each colum being the vector for each categori. this type of encoding gived the neural network better results.

## embedding categorical data
for categorical data we can preproccess the data frames before feedign it into the neural network. this can be filling out missing datas, forming catgeorical data into qontinuous, and normalize the data (subtract by the mean and divide by the standard deviation)

* mossinf data we replace it by the median and add colum telling if it was missin or not
* Normalization or all the other as well what ever you do to the training set has to be done to validation and test too.
* categorical to continues instead of having a vector where all values are 0 and one is 1 representing the category. Each categori has a floating number for every row and similar categories has high floating number at similar rows, so we can the capture similarities between the categorical data. these embeddings can then later be sendt into any neural network

the row holds information about things, and the more useful it for us if the rown hold information that is relevant for the domain.

the higher in order shcool, hogh school, uni the more of the rows are used to describe a value.
None: [0,0,0]
School: [-0.5, -0.5, 0]
Uni: [-0.25, -0.25, 0.5]

this information can be valuable fr the neulral network as it can see relations with categorical sub info that might realte to the final oucome.

this happens in a embedding layer that is used to extract useful information between layers. this can be done for a lot of things the word it self, the meta deta of the wors as length long job title may say somehting, subdomain as in email .org and .edu tells something about the person as salary maybe.

### tabular data
tabular data is data as its store din tables, csv files that are comma seperated. this data can be loaded in pyhtne with loead command combinde with tables

Table data = loadTable('data.csv', "header")
second argument is optional to tell it has an header.

## related works
entity embedding helps the neural network to generalize better when the data is sparse and statistics are unknown. Tus its helpfl when the data has a lots of features.

det ser ut som at etter identity embedding så lignr ikka bare vektorene på hverandre, men de peker samme retning ogsåå. som King - Man får lignene resultat som Queen - Woman. målet er å maksimere sannsyligheten for at wc er in kontekst med w der w of wc er to vectorer som representerer et ord.

p(wc|w) = exp(w*wc)/sum_i(w*wi) 

### single desision tree

can make decision on both categories and continous data, and also with bothh. The tree has a root node, internal nodes and leaf nodes

#### how to go from raw data to a desicion thee.
Decision tree are often used when we have categorical data as they do not convert them to continues.

1. need to figurare out which feature needs to be on top, we see how good they classify the heat diases alone.
2. Almost no feature is perfect so we calculate, one method used in gini
	1. calculate 1-(probability of correct yes)**2 - (prbability of wrong no)**2
	2. then calculate 1 -(prob wrong yes)**2 - (prob correct no)**2
	3. since persons in leaf nodes of yes and no, may not be equal wee need to multiply the gini impeurity with the number of people in a laf node divided with number of people on both leaf node to add a weight to the impeurity. then add them togheter
	4. after calcuculating gini impeurity for each feature we can pick the one with the lowest ot be the leaf node.
3. after pickeing the leaf nodes, we split the people in two leaf nodes, we need to calculate the ginie impeurity og the features on the poeple on each leaf node to see which feature should be check at that node.
4. at the end wee need to check of a inner nodes fetaure reduses the gini impeurity from the above inner node if it does we can add it if not its no point.
5. if numeric data sort the tree from lowest to highest, then calculate the average weigth for all adjacent patients, then calculate the gini impureity vale.

#### random forests

1. we make bootstrapped dataset where we pick ranodm sample drom the data, thereb we can pick the same sample mutiple times

2. pick random number of random features and makes a decision tree out of them 
3. then do the same severela times whch end up in many decition trees. 
4. For classification we send all the value throght all of the decition trees. This give better preditcion then single decision tree. and count how many gives one class and then pick the class most gave.

## Ensembled laerning
random forest is a example of ensembled leanirng as we make many classifiers and sent the data throtgh them. we create limited or man weak learners so togheter they can performe as a strong learner. We make multiple learniner useing the same learning algorithm. ensemble learning help minimize noise, bias and variance

## bagging and boosting
bootrstrap then pcik the average is calles bagging. in baggign all data sets or elements have the same chance of appearing in the new data set. for boosting the observation are weighted and the observation that wher misclassified will have higher weight i.e more chance of being picked. even thou both averege the final decision for baggign the averega e is equal while for boosting its weighted.

only boosting tryis to reduce the bias, but as wll bagging may solve the over fitting while boosting might increase it.

### gradiant boosted tree

## structured data
by structured data we mean a table format with features in colums and data examples in rows. here we meet the problem where continous variables can be representinted by real numbers while categorical needs to be represented either by integers, but as we know the integers dont tell the relations between the categories. These are called nominal numbers other time the integers can be properties of the catgeorical variables as for weeks, month etc.These are called  cardinam numbers or ordinal numbers but the mening might not be any more useful as jan and feb are close month but have dirrent ranges for days here jun is closer to jan.

common practise with tabular data is, logistic regression, random forrest or gradiant boosting machines. but the common wisom is worng neural netwoks are usefule with tabular data.  pinterest witc over which gave hthem better accuracy and less meintanence as less hand coded variables where needed.they repleced the gradiant boosting macines. they used it to figuare out what contenct to put on their homepage. xgboost software for gradian boosting

library to use naural networks with tabular data.

fast ai.
from fastai import *
from fast.tabular import *

assume data is in pandas dataframe. whoch is the standard format for tabular data in python.

need to tell it wat are your categorical variabels and what are continous

then we need to do something that a bit lik transform in computer vision. flip, r brighten or sharpen the image. here wee ned to normalize, fill missing values and categorify. instead of transforms we have processes. trasofrms are thing you do different going through. while processes is something you do one time ahaeas of time. before.
1. missin data: replace with median add nwe colum telling if it was missing.

## Entity embedding

1. we map each discrete variable to a vector
2. we have a onte-hot-encoded ata layer where we make each dicrete variable to be a vector with the length of the possible values for the discrete variable and pick one point to be the representation of the single var xi in the vector, this value is 1 in the vector while the others are 0.
3. the entity embedding layer takes inn this vectors and uouput a new vector, where aplha are the possible values for x_i X = sum_alpha(Wab*delta 


* We start with choosing a number for each categorival variabel value, as eye color blue = 1, green = 2 etc..
* Then we make these into one hot encoded vector which are vector with the lengt of number of values and where the index == value number there is a 1 and all other points on the vector is 0
* then the one-hot-encoded vector is multiplied with a number_value*D vector which hold the weight since the one-hot encoded vector only has one 1 and rest is zero it will pick one row from the number_val*D weights vector and this row is the representation of the that value we can choose how the lengt of the vector and values we want to represent the value with. commen 200 thousentd more give better acceasy and slower code
* After that we concateneate all the embedding layers and send them inro a dense layer. with backpropagation we change the weights in the embedding layer to and close values should have similar chamges.

still meant to be linears so wee ned to adda bias to the embedding matrix.

The middle layer can use relu as activation , while tha last uses sigmoid or softmax.
 
we can use sigmoid to get in range of sum number i.g 0-5. as sigmoid gives from 0-1 we can use sigmoid(x)*(max_score -min_score) + min score.

The D is the hyperparamter that needs to be predefined. in practice we define dimension based on experiments. the following guideline is used firste the more complex the more dimensions. if we hav esome idea of how many is needed we start with that if we dont have a clue we start with number of values -1.
