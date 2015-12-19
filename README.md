ml-decision-tree
=====

prerequisite
-----

1. basic python program experience
1. tree data structure
1. depth first search algorithm
1. basic knowledge about machine learning, knowing meaning of training, predict 

before digging into the source code
-----
you need to know some thing about decision tree background, 
if you are familiar with decision tree and know then meaning of
ID3 algorithm, C4.5, or CART, you can skip this part.

### what's decision tree, and decision learning
the picture below is a [decision tree](https://en.wikipedia.org/wiki/Decision_tree) showing survival of passengers 
on the Titanic, The figures under the leaves show the probability of survival and 
the percentage of observations in the leaf.
you can predict a person die or survived given this 
decision tree and this person's gender, age, and number of 
siblings/spouses aboard(sibsp).


for example, if given this decision tree, and given a person whose gender
 is female, age is 25, number of siblings/spouses aboard is 1, we can 
 predict that this person died during titanic trip.   

![a decision tree](./images/CART_tree_titanic_survivors.png)
 
Decision tree learning uses a decision tree as a predictive model 
which maps observations about an item to conclusions about the item's target value. 
It is one of the predictive modelling approaches used in statistics, data mining and machine learning. 
Tree models where the target variable can take a finite set of values are called classification trees. 
In these tree structures, leaves represent class labels and branches represent conjunctions of features 
that lead to those class labels. Decision trees where the target variable can take continuous values 
(typically real numbers) are called regression trees. you can know more from [wikipedia](https://en.wikipedia.org/wiki/Decision_tree_learning)

### kind of decision tree learning
Decision tree learning is the construction of a decision tree from class-labeled training tuples. 
A decision tree is a flow-chart-like structure, where each internal (non-leaf) node denotes a 
test on an attribute, each branch represents the outcome of a test, and each leaf (or terminal) 
node holds a class label. The top most node in a tree is the root node.

There are many specific decision-tree algorithms. Notable ones include:

- ID3 (Iterative Dichotomiser 3)
- C4.5 (successor of ID3)
- CART (Classification And Regression Tree), etc

ID3 and CART were invented independently at around the same time (between 1970 and 1980)[citation needed], yet follow a similar approach for learning decision tree from training tuples.
you can know more from [wikipedia](https://en.wikipedia.org/wiki/Decision_tree_learning)

know more about ID3 from [wikipedia](https://en.wikipedia.org/wiki/ID3_algorithm).

get a basic knowledge about C4.5 from [here](https://en.wikipedia.org/wiki/C4.5_algorithm#Improvements_from_ID.3_algorithm),

get more about C4.5 from this [tutorial](http://cis-linux1.temple.edu/~giorgio/cis587/readings/id3-c45.html)

get the difference between ID3 and C4.5 from this [paper](http://saiconference.com/Downloads/SpecialIssueNo10/Paper_3-A_comparative_study_of_decision_tree_ID3_and_C4.5.pdf) named "A comparative study of decision tree ID3 and C4. 5."
 by HSSINA, Badr, et al. published in International Journal of Advanced Computer Science and Applications 4.2 (2014).

get insight about CART decision tree from youtube video:

1. [(ML 2.1) Classification trees (CART)](https://www.youtube.com/watch?v=p17C9q2M00Q)
1. [(ML 2.2) Regression trees (CART)](https://www.youtube.com/watch?v=zvUOpbgtW3c)
1. [(ML 2.3) Growing a regression tree (CART)](https://www.youtube.com/watch?v=_RxqyvRK0Rw)
1. [(ML 2.4) Growing a classification tree (CART)](https://www.youtube.com/watch?v=S51plSJBC2g)
1. [(ML 2.5) Generalizations for trees (CART)](https://www.youtube.com/watch?v=UMtBWQ2m04g)



about the source code
-----

### data

the data from paper "Letter Recognition Using Holland-style Adaptive Classifiers".

The objective is to identify each of a large number of black-and-white rectangular pixel displays as one of the 26 capital letters in the English alphabet. The character images were based on 20 different fonts and each letter within these 20 fonts was randomly distorted to produce a file of 20,000 unique stimuli. Each stimulus was converted into 16 primitive numerical attributes (statistical moments and edge counts) which were then scaled to fit into a range of integer values from 0 through 15. We typically train on the first 16000 items and then use the resulting model to predict the letter category for the remaining 4000.

you can know more detail and download data from [uci machine learning repository](https://archive.ics.uci.edu/ml/datasets/Letter+Recognition)

### algorithm

the algorithm implemented here is a simple version of CART decision tree.

the code structure are learned from [DecisionTreeClassifier of scikit-learn](https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tree/tree.py)


how to run test code
-----

### run environment

1. windows or *nix system, tested on *nix system   
1. python version 2.7 or higher, tested at 2.7

`python decision_trees_sample.py`




### result

during the program, it takes 16000 instance as training data, it will 
output gini index value of each tree node during training time.
after training, it will output the prediction precision of 4000 instance 
of test data.

the program will consume your almost 1 minute time 

the result will look like this:

    param: 2.000000
    gini index 0.000000
    best_feature: 5
    param: 3.000000
    gini index 0.178571
    best_feature: 0
    param: 8.000000
    gini index 0.000000
    best_feature: 0
    param: 5.000000
    gini index 0.000000
    precision 3478 / 4000 = 0.869500
