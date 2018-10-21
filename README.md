
# Supervised Tweets Prediction

## Introduction

   This project is a small part of a bigger project from my University where the objective was to create an application (mobile and web), which would help people to combat mosquitos that are vectors of _Zika_ virus, reporting their focus (propitious places to reproduce).
   To make the platform more precisely, we collected data from Twitter (tweets through _keywords_) that could help us to point suspicous places in the map.

## Required background knowledge:

* Is good to already have knowledge:
 * Existing predictive models
 * Feature extraction
 * Model evaluation

### Know existing predictive models

I used some existing and available models on sklearn.

### Feature extraction

Consists in transforming arbitrary data, such as text or images, into numerical features usable for machine learning. In this project I just used three different Vectorizers to transforming text into usable features for the models.

### Model evaluation

Model Evaluation is a part of the model development process. It helps to find the best model that represents our data and how well the chosen model will work in the future.

## Objectives:

My objective is to classify if a tweet is useful or not, to help us to get information about mosquitos and _Zika_ virus.

This project consists in build a model to predict a labeled tweet.

* The labels are:

 * __Noise__: When tweets are useless in help to get information about mosquitos and _Zika_.
 
 * __Receiver__: When tweets can help to get information about mosquitos and _Zika_, but no so well.
 
 * __Provider__:When tweets are useful in help to get information about mosquitos and _Zika_.
 

### Starting to code:

_Importing libraries_:


```python
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import *
from sklearn.dummy import *
from sklearn.ensemble import *
from sklearn.neighbors import *
from sklearn.tree import *
from sklearn.svm import *
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import *
from sklearn.multiclass import *
import pandas as pd
import warnings
```

Let's ignore some warnings, about depracated stuff:


```python
warnings.filterwarnings("ignore")
```


Reading the dataset:



```python
data = pd.read_csv('/home/macdowell/workspace/classifying-tweets/tweets.csv')
```

Let's see the kind of data and your related types:


```python
print(data.columns)
print("---------------------------------------------")
print(data.dtypes)
```

    Index(['id', 'text', 'type'], dtype='object')
    ---------------------------------------------
    id      float64
    text     object
    type     object
    dtype: object


Our dataset have three columns and as we can see, they are with correct type. We don't need do any kind of cast.

For our abroad, we won't use the *id* of the tweets, so let's drop them:


```python
data = data.drop(columns='id')
```

Now I'm going to split our data set in two sets: *training set* and *test set*.
For this, I'll do it automaticly with method __train_test_split()__ from module sklearn.model_selection, where you choose by parameter: test_size and if want to shuffle dataset, then it returns a tuple: __train set, test set, train target set, test target set__.


```python
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['type'], test_size=0.3, shuffle=True)
```

Obs: Here we also could use __Cross-validation__ from sklearn.model_selection.KFold to split our train/test instances.

## Start to perform machine learning models

I created a method to run all algorithms with all Vectorizers once:


```python
def perform_models(classifiers, vectorizers, X_train, X_test, y_train, y_test):
    string = ''
    for classifier in classifiers:
        for vectorizer in vectorizers:
            string += classifier.__class__.__name__ + ' with ' + vectorizer.__class__.__name__

            # train
            vectorize_text_train = vectorizer.fit_transform(X_train)
            classifier.fit(vectorize_text_train, y_train)
            # score
            vectorize_text_test = vectorizer.transform(X_test)

            predicteds = classifier.predict(vectorize_text_test)
            precision, recall, f1, supp = precision_recall_fscore_support(y_test, predicteds,
                                                                          average='weighted')
            string += '. Precision: ' + str(precision) + '. Recall: ' + str(recall) + ' F1: ' + str(f1) + '\n'
            print(string)

```

Now calling the method:


```python
perform_models(
    [
        AdaBoostClassifier(),
        BaggingClassifier(),
        ExtraTreesClassifier(),
        GradientBoostingClassifier(),
        DecisionTreeClassifier(),
        DummyClassifier(),
        PassiveAggressiveClassifier(),
        RidgeClassifier(),
        RidgeClassifierCV(),
        SGDClassifier(),
        OneVsRestClassifier(SVC(kernel='linear')),
        OneVsRestClassifier(LogisticRegression()),
        KNeighborsClassifier(),
        BernoulliNB(),
        RandomForestClassifier(n_estimators=100, n_jobs=-1),

    ],
    [
        CountVectorizer(),
        TfidfVectorizer(),
        HashingVectorizer()
    ],
    X_train, X_test,
    y_train, y_test
)


```

    AdaBoostClassifier with CountVectorizer. Precision: 0.5834634305399451. Recall: 0.4537953795379538 F1: 0.42903875434487876
    
    AdaBoostClassifier with CountVectorizer. Precision: 0.5834634305399451. Recall: 0.4537953795379538 F1: 0.42903875434487876
    AdaBoostClassifier with TfidfVectorizer. Precision: 0.5806869525342334. Recall: 0.44554455445544555 F1: 0.42046311714183365
    
    AdaBoostClassifier with CountVectorizer. Precision: 0.5834634305399451. Recall: 0.4537953795379538 F1: 0.42903875434487876
    AdaBoostClassifier with TfidfVectorizer. Precision: 0.5806869525342334. Recall: 0.44554455445544555 F1: 0.42046311714183365
    AdaBoostClassifier with HashingVectorizer. Precision: 0.5829263192484055. Recall: 0.44884488448844884 F1: 0.4241358698109238
    
    AdaBoostClassifier with CountVectorizer. Precision: 0.5834634305399451. Recall: 0.4537953795379538 F1: 0.42903875434487876
    AdaBoostClassifier with TfidfVectorizer. Precision: 0.5806869525342334. Recall: 0.44554455445544555 F1: 0.42046311714183365
    AdaBoostClassifier with HashingVectorizer. Precision: 0.5829263192484055. Recall: 0.44884488448844884 F1: 0.4241358698109238
    BaggingClassifier with CountVectorizer. Precision: 0.7611005582948024. Recall: 0.7607260726072608 F1: 0.7423403707064592
    
    AdaBoostClassifier with CountVectorizer. Precision: 0.5834634305399451. Recall: 0.4537953795379538 F1: 0.42903875434487876
    AdaBoostClassifier with TfidfVectorizer. Precision: 0.5806869525342334. Recall: 0.44554455445544555 F1: 0.42046311714183365
    AdaBoostClassifier with HashingVectorizer. Precision: 0.5829263192484055. Recall: 0.44884488448844884 F1: 0.4241358698109238
    BaggingClassifier with CountVectorizer. Precision: 0.7611005582948024. Recall: 0.7607260726072608 F1: 0.7423403707064592
    BaggingClassifier with TfidfVectorizer. Precision: 0.7362386098553834. Recall: 0.7343234323432343 F1: 0.7297372051323762
    



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-22-9d5e95b46228> in <module>
         24     ],
         25     X_train, X_test,
    ---> 26     y_train, y_test
         27 )
         28 


    <ipython-input-21-2ededadc0de5> in perform_models(classifiers, vectorizers, X_train, X_test, y_train, y_test)
          7             # train
          8             vectorize_text_train = vectorizer.fit_transform(X_train)
    ----> 9             classifier.fit(vectorize_text_train, y_train)
         10             # score
         11             vectorize_text_test = vectorizer.transform(X_test)


    /usr/local/lib/python3.6/dist-packages/sklearn/ensemble/bagging.py in fit(self, X, y, sample_weight)
        242         self : object
        243         """
    --> 244         return self._fit(X, y, self.max_samples, sample_weight=sample_weight)
        245 
        246     def _fit(self, X, y, max_samples=None, max_depth=None, sample_weight=None):


    /usr/local/lib/python3.6/dist-packages/sklearn/ensemble/bagging.py in _fit(self, X, y, max_samples, max_depth, sample_weight)
        372                 total_n_estimators,
        373                 verbose=self.verbose)
    --> 374             for i in range(n_jobs))
        375 
        376         # Reduce


    /usr/local/lib/python3.6/dist-packages/sklearn/externals/joblib/parallel.py in __call__(self, iterable)
        981             # remaining jobs.
        982             self._iterating = False
    --> 983             if self.dispatch_one_batch(iterator):
        984                 self._iterating = self._original_iterator is not None
        985 


    /usr/local/lib/python3.6/dist-packages/sklearn/externals/joblib/parallel.py in dispatch_one_batch(self, iterator)
        823                 return False
        824             else:
    --> 825                 self._dispatch(tasks)
        826                 return True
        827 


    /usr/local/lib/python3.6/dist-packages/sklearn/externals/joblib/parallel.py in _dispatch(self, batch)
        780         with self._lock:
        781             job_idx = len(self._jobs)
    --> 782             job = self._backend.apply_async(batch, callback=cb)
        783             # A job can complete so quickly than its callback is
        784             # called before we get here, causing self._jobs to


    /usr/local/lib/python3.6/dist-packages/sklearn/externals/joblib/_parallel_backends.py in apply_async(self, func, callback)
        180     def apply_async(self, func, callback=None):
        181         """Schedule a func to be run"""
    --> 182         result = ImmediateResult(func)
        183         if callback:
        184             callback(result)


    /usr/local/lib/python3.6/dist-packages/sklearn/externals/joblib/_parallel_backends.py in __init__(self, batch)
        543         # Don't delay the application, to avoid keeping the input
        544         # arguments in memory
    --> 545         self.results = batch()
        546 
        547     def get(self):


    /usr/local/lib/python3.6/dist-packages/sklearn/externals/joblib/parallel.py in __call__(self)
        259         with parallel_backend(self._backend):
        260             return [func(*args, **kwargs)
    --> 261                     for func, args, kwargs in self.items]
        262 
        263     def __len__(self):


    /usr/local/lib/python3.6/dist-packages/sklearn/externals/joblib/parallel.py in <listcomp>(.0)
        259         with parallel_backend(self._backend):
        260             return [func(*args, **kwargs)
    --> 261                     for func, args, kwargs in self.items]
        262 
        263     def __len__(self):


    /usr/local/lib/python3.6/dist-packages/sklearn/ensemble/bagging.py in _parallel_build_estimators(n_estimators, ensemble, X, y, sample_weight, seeds, total_n_estimators, verbose)
        109                 curr_sample_weight[not_indices_mask] = 0
        110 
    --> 111             estimator.fit(X[:, features], y, sample_weight=curr_sample_weight)
        112 
        113         else:


    /usr/local/lib/python3.6/dist-packages/sklearn/tree/tree.py in fit(self, X, y, sample_weight, check_input, X_idx_sorted)
        797             sample_weight=sample_weight,
        798             check_input=check_input,
    --> 799             X_idx_sorted=X_idx_sorted)
        800         return self
        801 


    /usr/local/lib/python3.6/dist-packages/sklearn/tree/tree.py in fit(self, X, y, sample_weight, check_input, X_idx_sorted)
        363                                            min_impurity_split)
        364 
    --> 365         builder.build(self.tree_, X, y, sample_weight, X_idx_sorted)
        366 
        367         if self.n_outputs_ == 1:


    KeyboardInterrupt: 



As we can see, each model was performed with three different Vectorizer algorithms and outputed your related metrics scores.


## Conclusions
