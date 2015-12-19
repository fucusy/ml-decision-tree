# ml-decision-tree

## prerequisite

1. basic python program experience
1. tree data structure
1. depth first search algorithm
1. basic knowledge about machine learning, knowing meaning of training, predict 

##before digging into the source code
you need to know some thing about decision tree background, 
if you are familiar with decision tree and know then meaning of
ID3 algorithm, C4.5, or CART, you can skip this part.

### what's decision tree, and decision learning
the picture below is a decision tree showing survival of passengers 
on the Titanic, The figures under the leaves show the probability of survival and 
the percentage of observations in the leaf.
you can predict a person die or survived given this 
decision tree and this person's gender, age, and number of 
siblings/spouses aboard(sibsp).


for example, if given this decision tree, and given a person whose gender
 is female, age is 25, number of siblings/spouses aboard is 1, we can 
 predict that this person died during titanic trip.   

![a decision tree](./images/CART_tree_titanic_survivors.png)
 
 Decision tree learning uses a decision tree as a predictive model which maps observations about an item to conclusions about the item's target value. It is one of the predictive modelling approaches used in statistics, data mining and machine learning. Tree models where the target variable can take a finite set of values are called classification trees. In these tree structures, leaves represent class labels and branches represent conjunctions of features that lead to those class labels. Decision trees where the target variable can take continuous values (typically real numbers) are called regression trees.
### kind of decision tree learning
Decision tree learning is the construction of a decision tree from class-labeled training tuples. A decision tree is a flow-chart-like structure, where each internal (non-leaf) node denotes a test on an attribute, each branch represents the outcome of a test, and each leaf (or terminal) node holds a class label. The topmost node in a tree is the root node.

There are many specific decision-tree algorithms. Notable ones include:
ID3 (Iterative Dichotomiser 3)
C4.5 (successor of ID3)
CART (Classification And Regression Tree), etc
ID3 and CART were invented independently at around the same time (between 1970 and 1980)[citation needed], yet follow a similar approach for learning decision tree from training tuples.
## about the source code

### data

### algorithm

## how to run

## result