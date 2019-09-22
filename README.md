# AdaBoost
AdaBoost algorithm from scratch

### Constructing Adaboost algorithm

##### Boosting for binary classification (General Approach):

<img src="https://render.githubusercontent.com/render/math?math=X^l = (x_i, y_i)_{i=1}^l"> - training sample

<img src="https://render.githubusercontent.com/render/math?math=a(x)=C(b(x))"> - algorithm, where

<img src="https://render.githubusercontent.com/render/math?math=b: X \rightarrow R"> - weak learner, R is an estimate, which is a real value, e.g. probability of belonging to certain class

<img src="https://render.githubusercontent.com/render/math?math=C: R \rightarrow Y"> - desicion rule, that converts estimate to discrete class label, e.g. <img src="https://render.githubusercontent.com/render/math?math=C(b(x)) = sign(b(x))">

boosting is a composition of <img src="https://render.githubusercontent.com/render/math?math=b_1 ... b_T"> weak learners s.t.

<img src="https://render.githubusercontent.com/render/math?math=a(x) = C(F(b_1(x) ... b_T(x))">, where

<img src="https://render.githubusercontent.com/render/math?math=F: R^T \rightarrow R"> - converts vector of estimates into one estimate.



Examples of F:

1. Simple voting:

<img src="https://render.githubusercontent.com/render/math?math=F(b_1(x) ... b_T(x) = \frac{1}{T} \sum_{t=1}^T b_t(x), \space x \in X">

1. Weighted voting:

<img src="https://render.githubusercontent.com/render/math?math=F(b_1(x) ... b_T(x) = \frac{1}{T} \sum_{t=1}^T \alpha_t b_t(x), \space x \in X, \space \alpha \in R">

For construction of AdaBoost classifier we will use weighted voting.


***


##### AdaBoost Algorithm:

<img src="https://render.githubusercontent.com/render/math?math=Y = \{+1, -1\}, \space b_t \rightarrow \{+1, -1\}, \space C(b) = sign(b(x))">

<img src="https://render.githubusercontent.com/render/math?math=a(x) = sign(\sum_{t=1}^T \alpha_t b_t(x))">

for b_t we will use decision stump which is the one-level decision tree.

At this step our loss fuction is a treshhold function that count number of misclassified samples:

<img src="https://render.githubusercontent.com/render/math?math=Q_t = \sum_{i=1}^l \underbrace{\big[y_i \sum_{i=1}^T \alpha_t b_t(x_i) < 0 \big]}_{M_i}">


We can't effectively optimize treshold function, so we must introduce two heurisitcs:
 1. First of all, we will use smooth function to majoritaze threshold loss, in case of AdaBoost this smooth function is exponential: <img src="https://render.githubusercontent.com/render/math?math=[M < 0] \leq e^{-M}"> 
 
 2. Secondly, we will turn one optimization problem into T sequential ones, thus <img src="https://render.githubusercontent.com/render/math?math=\alpha_1b_1(x)...\alpha_{t-1}b_{t-1}(x)"> are fixed when we add <img src="https://render.githubusercontent.com/render/math?math=\alpha_tb_t(x)">.
 


Plugging exponent into <img src="https://render.githubusercontent.com/render/math?math=Q_t">:

<img src="https://render.githubusercontent.com/render/math?math=Q_T \leq \tilde Q_T = \sum_{i=1}^l \underbrace{exp\big(-y_i\sum_{t=1}^{T-1}\alpha_tb_t(x_i)\big)}_{w_i}exp(-y_i\alpha_Tb_T(x_i))">

Let's interpret <img src="https://render.githubusercontent.com/render/math?math=w_i">:

If <img src="https://render.githubusercontent.com/render/math?math=w_i"> is big is big then accumulated in <img src="https://render.githubusercontent.com/render/math?math=T">-1 iterations <img src="https://render.githubusercontent.com/render/math?math=M_i"> negative and large in absolute terms, which means <img src="https://render.githubusercontent.com/render/math?math=i">'th object is difficult for classification. Thus when we run new alogrithm <img src="https://render.githubusercontent.com/render/math?math=b_T"> we prioritize those object that were difficult for classification on <img src="https://render.githubusercontent.com/render/math?math=T">-1 steps. That's where the name of algorithm comes from, adaptive means that on each iteration new algorithm takes into account what objects were missclassified.

If we missclassify the object too often its weight become very large and otherwise if we always classify the object correctly weight become very small, this disproportion can cause numerical instability and it is recommended to normalize weights on each iteration:

<img src="https://render.githubusercontent.com/render/math?math=\tilde w_i = w_i / \sum_{j=1}^N w_j">

<img src="https://render.githubusercontent.com/render/math?math=\tilde W^l = (\tilde w_1 ... \tilde w_l)">

Next, we need to introduce two metrics:

<img src="https://render.githubusercontent.com/render/math?math=P(b,\tilde W^l)"> and <img src="https://render.githubusercontent.com/render/math?math=N(b,\tilde W^l)"> which are the weighted number of correctly and incorrectly classified samples.



At 1995, creators of AdaBoost algorithm Freund and Shapire presented the following theorem in their work:

Let for any <img src="https://render.githubusercontent.com/render/math?math=\tilde W^l"> there is a <img src="https://render.githubusercontent.com/render/math?math=b \in B"> s.t. <img src="https://render.githubusercontent.com/render/math?math=N(b, \tilde W^l) < \frac{1}{2}">, i.e. <img src="https://render.githubusercontent.com/render/math?math=b"> is atleast slightly better then random guessing, then <img src="https://render.githubusercontent.com/render/math?math=\tilde Q_T"> is minimzed when:

<img src="https://render.githubusercontent.com/render/math?math=b_T=\arg\min_{b\inB}N(b,\tilde W^l)">

<img src="https://render.githubusercontent.com/render/math?math=\alpha_T=\frac{1}{2}ln\big(\frac{1-N(b,\tilde W^l)}{N(b,\tilde W^l)}\big)">

where <img src="https://render.githubusercontent.com/render/math?math=\alpha_T"> is found from the equation <img src="https://render.githubusercontent.com/render/math?math=\frac{\partial \tilde Q_t}{\partial \alpha_t} = 0">



Now we can write the whole AdaBoost algorithm:

In: <img src="https://render.githubusercontent.com/render/math?math=X^l">, parameter <img src="https://render.githubusercontent.com/render/math?math=T">

Out: <img src="https://render.githubusercontent.com/render/math?math=\alpha_tb_t">, <img src="https://render.githubusercontent.com/render/math?math=t=1,...,T">

<img src="https://render.githubusercontent.com/render/math?math=1">. Initialize weights:
    <img src="https://render.githubusercontent.com/render/math?math=w_i:=\frac{1}{l}$, $i=1,...,l">

2. for each t,...,T:
    
    <img src="https://render.githubusercontent.com/render/math?math=3">. <img src="https://render.githubusercontent.com/render/math?math=b_t := \arg\min_{b} N(b, \tilde W^l)">
    
    <img src="https://render.githubusercontent.com/render/math?math=4">. <img src="https://render.githubusercontent.com/render/math?math=\alpha_t := \frac{1}{2} ln\big(\frac{1-N(b, \tilde W^l)}{N(b, \tilde W^l)}\big)">  

    <img src="https://render.githubusercontent.com/render/math?math=5">. <img src="https://render.githubusercontent.com/render/math?math=w_i := w_i exp(-y_i \alpha_t, b_t(x_i))">, <img src="https://render.githubusercontent.com/render/math?math=i=1,...,l">

    <img src="https://render.githubusercontent.com/render/math?math=6">. Normalize weights: <img src="https://render.githubusercontent.com/render/math?math=\tilde w_i = w_i / \sum_{j=1}^N w_j">, <img src="https://render.githubusercontent.com/render/math?math=i=1,...,l">


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn import tree
```


```python
ds = pd.read_csv('toy_ds.csv')
X = np.array(ds.iloc[:,:2])
y = np.array(ds.iloc[:,2])
```


```python
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

plt.scatter(X[:,0], X[:,1], c=y.tolist(), cmap=cm_bright)
plt.grid()
plt.show()
```


![png](output_4_0.png)



```python
# Example of decision stump that we are going to use as weak learner

clf = DTC(max_depth=1, max_leaf_nodes=2)
clf.fit(X, y)
tree.plot_tree(clf)
```




    [Text(167.4, 163.07999999999998, 'X[1] <= 0.639\nentropy = 0.5\nsamples = 40\nvalue = [20, 20]'),
     Text(83.7, 54.360000000000014, 'entropy = 0.436\nsamples = 28\nvalue = [9, 19]'),
     Text(251.10000000000002, 54.360000000000014, 'entropy = 0.153\nsamples = 12\nvalue = [11, 1]')]




![png](output_5_1.png)



```python
x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .11
y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .11
xx, yy = np.meshgrid( np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100) )

Xfull = np.c_[xx.ravel(), yy.ravel()]
zz = np.array( [clf.predict_proba(Xfull)[:,1]] )
Z = zz.reshape(xx.shape)

plt.contourf(xx, yy, Z, 4, cmap='RdBu', alpha=.5)
plt.contour(xx, yy, Z, 2, cmap='RdBu')
plt.scatter(X[:,0],X[:,1], c=y, cmap = cm_bright)
plt.xlabel('$X_1$')
plt.ylabel('$X_2$')
plt.show()
```


![png](output_6_0.png)


Function that runs AdaBoost algo on given sample:


```python
def adaboost(X, y, T):
    '''
    error = estimator error
    alpha = estimator weight
    weights = sample weights
    '''
    estimator_list, y_predict_list, estimator_error_list, estimator_weight_list, sample_weight_list = [],[],[],[],[]
    weights = np.ones(len(y))/len(y)
    sample_weight_list.append(weights.copy())
    #print(weights)
    
    for t in range(T):
        
        clf = DTC(max_depth=1, max_leaf_nodes=2)
        clf.fit(X, y, sample_weight=weights)
        y_predict = clf.predict(X)

        incorrect = (y_predict!=y)
        error = (np.average(incorrect, weights=weights))/sum(weights)
        alpha = (1/2)*np.log((1-error) / error)
        weights *= (np.e**(-(alpha*y*y_predict)))/sum(weights)
        #print('iter:',t, weights)
        
        estimator_list.append(clf)
        y_predict_list.append(y_predict.copy())
        estimator_error_list.append(error.copy())
        estimator_weight_list.append(alpha.copy())
        sample_weight_list.append(weights.copy())
        #print(sample_weight_list)
        
    estimator_list = np.asarray(estimator_list)
    y_predict_list = np.asarray(y_predict_list)
    estimator_error_list = np.asarray(estimator_error_list)
    estimator_weight_list = np.asarray(estimator_weight_list)
    sample_weight_list = np.asarray(sample_weight_list)
    
    preds = np.array([np.sign((y_predict_list[:,i] * estimator_weight_list).sum()) for i in range(len(y))])
    #print('Accuracy:', (preds==y).sum()/len(y))
   
    return estimator_list, estimator_weight_list, sample_weight_list
```


```python
estimator_list, estimator_weight_list, sample_weight_list = adaboost(X, y, 100)
```

Function that plots decison boundary on every iteration:


```python
def decision_boundary(adaboost, X, y):
    '''
    function plots decision boundary 
    for every iteration of adaboost algo 
    '''
    
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid( np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100) )
    Xfull = np.c_[xx.ravel(), yy.ravel()]
    
    estimator_list, estimator_weight_list, sample_weight_list = adaboost(X, y, 10)
    
    fig = plt.figure(figsize = (14,14))
    
    for est, sample_weights, m in zip(estimator_list, sample_weight_list, range(0, 10)):
        
        zz = np.array( [est.predict_proba(Xfull)[:,1]] )
        Z = zz.reshape(xx.shape)
        
        fig.add_subplot(4,3,m+1)
        plt.subplots_adjust(hspace = 0.3)
        ax = plt.gca()
        ax.contourf(xx, yy, Z, 2, cmap='RdBu', alpha=.5)
        ax.contour(xx, yy, Z,  2, cmap='RdBu')
        ax.scatter(X[:,0],X[:,1], c = y, s=sample_weights*1000, cmap = cm_bright)
        ax.title.set_text('Weak Learner '+str(m+1))
        ax.set_xlabel('$X_1$')
        ax.set_ylabel('$X_2$')
```


```python
# Let's see what happens on each iteration

decision_boundary(adaboost, X, y)
```


![png](output_12_0.png)


Function that plots final decision boundary:


```python
def final_boundary(estimators, estimator_weights, X, y):
    '''
    This function plots final decision boundary,
    i.e. boundary produced by the ensemble of
    weak learners
    '''
    
    def adaboost_classify_point(X, estimators, estimator_weights):
        '''
        Return classification prediction for a 
        given point X and a previously fitted adaboost
        '''
        weak_preds = np.asarray( [ a*e.predict(X) for a, e in zip(estimator_weights, estimators) ]  
                                / estimator_weights.sum() )
        preds = np.sign(weak_preds.sum(axis=0))
        return preds
        
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid( np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100) )
    Xfull = np.c_[xx.ravel(), yy.ravel()]
    
    zz = np.array( adaboost_classify_point(Xfull, estimator_list, estimator_weight_list) )
    Z = zz.reshape(xx.shape)
    
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.gca()
    ax.contourf(xx, yy, Z, 2, cmap='RdBu', alpha=.5)
    ax.contour(xx, yy, Z,  2, cmap='RdBu')
    ax.scatter(X[:,0],X[:,1], c=y, cmap = cm_bright)
    ax.set_xlabel('$X_1$')
    ax.set_ylabel('$X_2$')
```


```python
final_boundary(estimator_list, estimator_weight_list, X, y)
```


![png](output_15_0.png)

