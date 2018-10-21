
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

    AdaBoostClassifier with CountVectorizer. Precision: 0.5624601933877599. Recall: 0.641914191419142 F1: 0.5406036589115659
    AdaBoostClassifier with TfidfVectorizer. Precision: 0.5339725237583999. Recall: 0.6320132013201321 F1: 0.5699842045648011
    AdaBoostClassifier with HashingVectorizer. Precision: 0.53907862696382. Recall: 0.636963696369637 F1: 0.5593764668638859
    
    AdaBoostClassifier with CountVectorizer. Precision: 0.5624601933877599. Recall: 0.641914191419142 F1: 0.5406036589115659
    AdaBoostClassifier with TfidfVectorizer. Precision: 0.5339725237583999. Recall: 0.6320132013201321 F1: 0.5699842045648011
    AdaBoostClassifier with HashingVectorizer. Precision: 0.53907862696382. Recall: 0.636963696369637 F1: 0.5593764668638859
    BaggingClassifier with CountVectorizer. Precision: 0.7669478999181969. Recall: 0.7557755775577558 F1: 0.7322443282888413
    BaggingClassifier with TfidfVectorizer. Precision: 0.7137424551617745. Recall: 0.7227722772277227 F1: 0.7128632696988642
    BaggingClassifier with HashingVectorizer. Precision: 0.7403858272509948. Recall: 0.7343234323432343 F1: 0.7135123764608311
    
    AdaBoostClassifier with CountVectorizer. Precision: 0.5624601933877599. Recall: 0.641914191419142 F1: 0.5406036589115659
    AdaBoostClassifier with TfidfVectorizer. Precision: 0.5339725237583999. Recall: 0.6320132013201321 F1: 0.5699842045648011
    AdaBoostClassifier with HashingVectorizer. Precision: 0.53907862696382. Recall: 0.636963696369637 F1: 0.5593764668638859
    BaggingClassifier with CountVectorizer. Precision: 0.7669478999181969. Recall: 0.7557755775577558 F1: 0.7322443282888413
    BaggingClassifier with TfidfVectorizer. Precision: 0.7137424551617745. Recall: 0.7227722772277227 F1: 0.7128632696988642
    BaggingClassifier with HashingVectorizer. Precision: 0.7403858272509948. Recall: 0.7343234323432343 F1: 0.7135123764608311
    ExtraTreesClassifier with CountVectorizer. Precision: 0.7549552926895123. Recall: 0.7392739273927392 F1: 0.7160856739271999
    ExtraTreesClassifier with TfidfVectorizer. Precision: 0.7431510883398655. Recall: 0.731023102310231 F1: 0.7090809574438138
    ExtraTreesClassifier with HashingVectorizer. Precision: 0.7463822516886578. Recall: 0.7244224422442245 F1: 0.6877920606866322
    
    AdaBoostClassifier with CountVectorizer. Precision: 0.5624601933877599. Recall: 0.641914191419142 F1: 0.5406036589115659
    AdaBoostClassifier with TfidfVectorizer. Precision: 0.5339725237583999. Recall: 0.6320132013201321 F1: 0.5699842045648011
    AdaBoostClassifier with HashingVectorizer. Precision: 0.53907862696382. Recall: 0.636963696369637 F1: 0.5593764668638859
    BaggingClassifier with CountVectorizer. Precision: 0.7669478999181969. Recall: 0.7557755775577558 F1: 0.7322443282888413
    BaggingClassifier with TfidfVectorizer. Precision: 0.7137424551617745. Recall: 0.7227722772277227 F1: 0.7128632696988642
    BaggingClassifier with HashingVectorizer. Precision: 0.7403858272509948. Recall: 0.7343234323432343 F1: 0.7135123764608311
    ExtraTreesClassifier with CountVectorizer. Precision: 0.7549552926895123. Recall: 0.7392739273927392 F1: 0.7160856739271999
    ExtraTreesClassifier with TfidfVectorizer. Precision: 0.7431510883398655. Recall: 0.731023102310231 F1: 0.7090809574438138
    ExtraTreesClassifier with HashingVectorizer. Precision: 0.7463822516886578. Recall: 0.7244224422442245 F1: 0.6877920606866322
    GradientBoostingClassifier with CountVectorizer. Precision: 0.7501302369042875. Recall: 0.7392739273927392 F1: 0.7154879043418036
    GradientBoostingClassifier with TfidfVectorizer. Precision: 0.7369080060565416. Recall: 0.7392739273927392 F1: 0.7272573566366475
    GradientBoostingClassifier with HashingVectorizer. Precision: 0.7634673319119184. Recall: 0.7524752475247525 F1: 0.7308409367026466
    
    AdaBoostClassifier with CountVectorizer. Precision: 0.5624601933877599. Recall: 0.641914191419142 F1: 0.5406036589115659
    AdaBoostClassifier with TfidfVectorizer. Precision: 0.5339725237583999. Recall: 0.6320132013201321 F1: 0.5699842045648011
    AdaBoostClassifier with HashingVectorizer. Precision: 0.53907862696382. Recall: 0.636963696369637 F1: 0.5593764668638859
    BaggingClassifier with CountVectorizer. Precision: 0.7669478999181969. Recall: 0.7557755775577558 F1: 0.7322443282888413
    BaggingClassifier with TfidfVectorizer. Precision: 0.7137424551617745. Recall: 0.7227722772277227 F1: 0.7128632696988642
    BaggingClassifier with HashingVectorizer. Precision: 0.7403858272509948. Recall: 0.7343234323432343 F1: 0.7135123764608311
    ExtraTreesClassifier with CountVectorizer. Precision: 0.7549552926895123. Recall: 0.7392739273927392 F1: 0.7160856739271999
    ExtraTreesClassifier with TfidfVectorizer. Precision: 0.7431510883398655. Recall: 0.731023102310231 F1: 0.7090809574438138
    ExtraTreesClassifier with HashingVectorizer. Precision: 0.7463822516886578. Recall: 0.7244224422442245 F1: 0.6877920606866322
    GradientBoostingClassifier with CountVectorizer. Precision: 0.7501302369042875. Recall: 0.7392739273927392 F1: 0.7154879043418036
    GradientBoostingClassifier with TfidfVectorizer. Precision: 0.7369080060565416. Recall: 0.7392739273927392 F1: 0.7272573566366475
    GradientBoostingClassifier with HashingVectorizer. Precision: 0.7634673319119184. Recall: 0.7524752475247525 F1: 0.7308409367026466
    DecisionTreeClassifier with CountVectorizer. Precision: 0.735328721362705. Recall: 0.735973597359736 F1: 0.7305161134303999
    DecisionTreeClassifier with TfidfVectorizer. Precision: 0.698245635113071. Recall: 0.698019801980198 F1: 0.6964044465362744
    DecisionTreeClassifier with HashingVectorizer. Precision: 0.7126253074833959. Recall: 0.7211221122112211 F1: 0.7117640914702404
    
    AdaBoostClassifier with CountVectorizer. Precision: 0.5624601933877599. Recall: 0.641914191419142 F1: 0.5406036589115659
    AdaBoostClassifier with TfidfVectorizer. Precision: 0.5339725237583999. Recall: 0.6320132013201321 F1: 0.5699842045648011
    AdaBoostClassifier with HashingVectorizer. Precision: 0.53907862696382. Recall: 0.636963696369637 F1: 0.5593764668638859
    BaggingClassifier with CountVectorizer. Precision: 0.7669478999181969. Recall: 0.7557755775577558 F1: 0.7322443282888413
    BaggingClassifier with TfidfVectorizer. Precision: 0.7137424551617745. Recall: 0.7227722772277227 F1: 0.7128632696988642
    BaggingClassifier with HashingVectorizer. Precision: 0.7403858272509948. Recall: 0.7343234323432343 F1: 0.7135123764608311
    ExtraTreesClassifier with CountVectorizer. Precision: 0.7549552926895123. Recall: 0.7392739273927392 F1: 0.7160856739271999
    ExtraTreesClassifier with TfidfVectorizer. Precision: 0.7431510883398655. Recall: 0.731023102310231 F1: 0.7090809574438138
    ExtraTreesClassifier with HashingVectorizer. Precision: 0.7463822516886578. Recall: 0.7244224422442245 F1: 0.6877920606866322
    GradientBoostingClassifier with CountVectorizer. Precision: 0.7501302369042875. Recall: 0.7392739273927392 F1: 0.7154879043418036
    GradientBoostingClassifier with TfidfVectorizer. Precision: 0.7369080060565416. Recall: 0.7392739273927392 F1: 0.7272573566366475
    GradientBoostingClassifier with HashingVectorizer. Precision: 0.7634673319119184. Recall: 0.7524752475247525 F1: 0.7308409367026466
    DecisionTreeClassifier with CountVectorizer. Precision: 0.735328721362705. Recall: 0.735973597359736 F1: 0.7305161134303999
    DecisionTreeClassifier with TfidfVectorizer. Precision: 0.698245635113071. Recall: 0.698019801980198 F1: 0.6964044465362744
    DecisionTreeClassifier with HashingVectorizer. Precision: 0.7126253074833959. Recall: 0.7211221122112211 F1: 0.7117640914702404
    DummyClassifier with CountVectorizer. Precision: 0.4540685703626661. Recall: 0.46864686468646866 F1: 0.46081773101695783
    DummyClassifier with TfidfVectorizer. Precision: 0.49890161894036195. Recall: 0.5033003300330033 F1: 0.5007505170128534
    DummyClassifier with HashingVectorizer. Precision: 0.42700854957744705. Recall: 0.44224422442244227 F1: 0.4332406915254077
    
    AdaBoostClassifier with CountVectorizer. Precision: 0.5624601933877599. Recall: 0.641914191419142 F1: 0.5406036589115659
    AdaBoostClassifier with TfidfVectorizer. Precision: 0.5339725237583999. Recall: 0.6320132013201321 F1: 0.5699842045648011
    AdaBoostClassifier with HashingVectorizer. Precision: 0.53907862696382. Recall: 0.636963696369637 F1: 0.5593764668638859
    BaggingClassifier with CountVectorizer. Precision: 0.7669478999181969. Recall: 0.7557755775577558 F1: 0.7322443282888413
    BaggingClassifier with TfidfVectorizer. Precision: 0.7137424551617745. Recall: 0.7227722772277227 F1: 0.7128632696988642
    BaggingClassifier with HashingVectorizer. Precision: 0.7403858272509948. Recall: 0.7343234323432343 F1: 0.7135123764608311
    ExtraTreesClassifier with CountVectorizer. Precision: 0.7549552926895123. Recall: 0.7392739273927392 F1: 0.7160856739271999
    ExtraTreesClassifier with TfidfVectorizer. Precision: 0.7431510883398655. Recall: 0.731023102310231 F1: 0.7090809574438138
    ExtraTreesClassifier with HashingVectorizer. Precision: 0.7463822516886578. Recall: 0.7244224422442245 F1: 0.6877920606866322
    GradientBoostingClassifier with CountVectorizer. Precision: 0.7501302369042875. Recall: 0.7392739273927392 F1: 0.7154879043418036
    GradientBoostingClassifier with TfidfVectorizer. Precision: 0.7369080060565416. Recall: 0.7392739273927392 F1: 0.7272573566366475
    GradientBoostingClassifier with HashingVectorizer. Precision: 0.7634673319119184. Recall: 0.7524752475247525 F1: 0.7308409367026466
    DecisionTreeClassifier with CountVectorizer. Precision: 0.735328721362705. Recall: 0.735973597359736 F1: 0.7305161134303999
    DecisionTreeClassifier with TfidfVectorizer. Precision: 0.698245635113071. Recall: 0.698019801980198 F1: 0.6964044465362744
    DecisionTreeClassifier with HashingVectorizer. Precision: 0.7126253074833959. Recall: 0.7211221122112211 F1: 0.7117640914702404
    DummyClassifier with CountVectorizer. Precision: 0.4540685703626661. Recall: 0.46864686468646866 F1: 0.46081773101695783
    DummyClassifier with TfidfVectorizer. Precision: 0.49890161894036195. Recall: 0.5033003300330033 F1: 0.5007505170128534
    DummyClassifier with HashingVectorizer. Precision: 0.42700854957744705. Recall: 0.44224422442244227 F1: 0.4332406915254077
    PassiveAggressiveClassifier with CountVectorizer. Precision: 0.7465797901302191. Recall: 0.7508250825082509 F1: 0.7438667084320119
    PassiveAggressiveClassifier with TfidfVectorizer. Precision: 0.7467228883856221. Recall: 0.7508250825082509 F1: 0.7452169407498447
    PassiveAggressiveClassifier with HashingVectorizer. Precision: 0.7462649540850699. Recall: 0.7442244224422442 F1: 0.7294605227113802
    
    AdaBoostClassifier with CountVectorizer. Precision: 0.5624601933877599. Recall: 0.641914191419142 F1: 0.5406036589115659
    AdaBoostClassifier with TfidfVectorizer. Precision: 0.5339725237583999. Recall: 0.6320132013201321 F1: 0.5699842045648011
    AdaBoostClassifier with HashingVectorizer. Precision: 0.53907862696382. Recall: 0.636963696369637 F1: 0.5593764668638859
    BaggingClassifier with CountVectorizer. Precision: 0.7669478999181969. Recall: 0.7557755775577558 F1: 0.7322443282888413
    BaggingClassifier with TfidfVectorizer. Precision: 0.7137424551617745. Recall: 0.7227722772277227 F1: 0.7128632696988642
    BaggingClassifier with HashingVectorizer. Precision: 0.7403858272509948. Recall: 0.7343234323432343 F1: 0.7135123764608311
    ExtraTreesClassifier with CountVectorizer. Precision: 0.7549552926895123. Recall: 0.7392739273927392 F1: 0.7160856739271999
    ExtraTreesClassifier with TfidfVectorizer. Precision: 0.7431510883398655. Recall: 0.731023102310231 F1: 0.7090809574438138
    ExtraTreesClassifier with HashingVectorizer. Precision: 0.7463822516886578. Recall: 0.7244224422442245 F1: 0.6877920606866322
    GradientBoostingClassifier with CountVectorizer. Precision: 0.7501302369042875. Recall: 0.7392739273927392 F1: 0.7154879043418036
    GradientBoostingClassifier with TfidfVectorizer. Precision: 0.7369080060565416. Recall: 0.7392739273927392 F1: 0.7272573566366475
    GradientBoostingClassifier with HashingVectorizer. Precision: 0.7634673319119184. Recall: 0.7524752475247525 F1: 0.7308409367026466
    DecisionTreeClassifier with CountVectorizer. Precision: 0.735328721362705. Recall: 0.735973597359736 F1: 0.7305161134303999
    DecisionTreeClassifier with TfidfVectorizer. Precision: 0.698245635113071. Recall: 0.698019801980198 F1: 0.6964044465362744
    DecisionTreeClassifier with HashingVectorizer. Precision: 0.7126253074833959. Recall: 0.7211221122112211 F1: 0.7117640914702404
    DummyClassifier with CountVectorizer. Precision: 0.4540685703626661. Recall: 0.46864686468646866 F1: 0.46081773101695783
    DummyClassifier with TfidfVectorizer. Precision: 0.49890161894036195. Recall: 0.5033003300330033 F1: 0.5007505170128534
    DummyClassifier with HashingVectorizer. Precision: 0.42700854957744705. Recall: 0.44224422442244227 F1: 0.4332406915254077
    PassiveAggressiveClassifier with CountVectorizer. Precision: 0.7465797901302191. Recall: 0.7508250825082509 F1: 0.7438667084320119
    PassiveAggressiveClassifier with TfidfVectorizer. Precision: 0.7467228883856221. Recall: 0.7508250825082509 F1: 0.7452169407498447
    PassiveAggressiveClassifier with HashingVectorizer. Precision: 0.7462649540850699. Recall: 0.7442244224422442 F1: 0.7294605227113802
    RidgeClassifier with CountVectorizer. Precision: 0.7508635193530426. Recall: 0.7524752475247525 F1: 0.7403578708823831
    RidgeClassifier with TfidfVectorizer. Precision: 0.7805865615724012. Recall: 0.7755775577557755 F1: 0.7649945842173317
    RidgeClassifier with HashingVectorizer. Precision: 0.7799049672409102. Recall: 0.7557755775577558 F1: 0.7304898640208674
    
    AdaBoostClassifier with CountVectorizer. Precision: 0.5624601933877599. Recall: 0.641914191419142 F1: 0.5406036589115659
    AdaBoostClassifier with TfidfVectorizer. Precision: 0.5339725237583999. Recall: 0.6320132013201321 F1: 0.5699842045648011
    AdaBoostClassifier with HashingVectorizer. Precision: 0.53907862696382. Recall: 0.636963696369637 F1: 0.5593764668638859
    BaggingClassifier with CountVectorizer. Precision: 0.7669478999181969. Recall: 0.7557755775577558 F1: 0.7322443282888413
    BaggingClassifier with TfidfVectorizer. Precision: 0.7137424551617745. Recall: 0.7227722772277227 F1: 0.7128632696988642
    BaggingClassifier with HashingVectorizer. Precision: 0.7403858272509948. Recall: 0.7343234323432343 F1: 0.7135123764608311
    ExtraTreesClassifier with CountVectorizer. Precision: 0.7549552926895123. Recall: 0.7392739273927392 F1: 0.7160856739271999
    ExtraTreesClassifier with TfidfVectorizer. Precision: 0.7431510883398655. Recall: 0.731023102310231 F1: 0.7090809574438138
    ExtraTreesClassifier with HashingVectorizer. Precision: 0.7463822516886578. Recall: 0.7244224422442245 F1: 0.6877920606866322
    GradientBoostingClassifier with CountVectorizer. Precision: 0.7501302369042875. Recall: 0.7392739273927392 F1: 0.7154879043418036
    GradientBoostingClassifier with TfidfVectorizer. Precision: 0.7369080060565416. Recall: 0.7392739273927392 F1: 0.7272573566366475
    GradientBoostingClassifier with HashingVectorizer. Precision: 0.7634673319119184. Recall: 0.7524752475247525 F1: 0.7308409367026466
    DecisionTreeClassifier with CountVectorizer. Precision: 0.735328721362705. Recall: 0.735973597359736 F1: 0.7305161134303999
    DecisionTreeClassifier with TfidfVectorizer. Precision: 0.698245635113071. Recall: 0.698019801980198 F1: 0.6964044465362744
    DecisionTreeClassifier with HashingVectorizer. Precision: 0.7126253074833959. Recall: 0.7211221122112211 F1: 0.7117640914702404
    DummyClassifier with CountVectorizer. Precision: 0.4540685703626661. Recall: 0.46864686468646866 F1: 0.46081773101695783
    DummyClassifier with TfidfVectorizer. Precision: 0.49890161894036195. Recall: 0.5033003300330033 F1: 0.5007505170128534
    DummyClassifier with HashingVectorizer. Precision: 0.42700854957744705. Recall: 0.44224422442244227 F1: 0.4332406915254077
    PassiveAggressiveClassifier with CountVectorizer. Precision: 0.7465797901302191. Recall: 0.7508250825082509 F1: 0.7438667084320119
    PassiveAggressiveClassifier with TfidfVectorizer. Precision: 0.7467228883856221. Recall: 0.7508250825082509 F1: 0.7452169407498447
    PassiveAggressiveClassifier with HashingVectorizer. Precision: 0.7462649540850699. Recall: 0.7442244224422442 F1: 0.7294605227113802
    RidgeClassifier with CountVectorizer. Precision: 0.7508635193530426. Recall: 0.7524752475247525 F1: 0.7403578708823831
    RidgeClassifier with TfidfVectorizer. Precision: 0.7805865615724012. Recall: 0.7755775577557755 F1: 0.7649945842173317
    RidgeClassifier with HashingVectorizer. Precision: 0.7799049672409102. Recall: 0.7557755775577558 F1: 0.7304898640208674
    RidgeClassifierCV with CountVectorizer. Precision: 0.7569831829763922. Recall: 0.7425742574257426 F1: 0.7184048954830016
    RidgeClassifierCV with TfidfVectorizer. Precision: 0.7752205496206167. Recall: 0.764026402640264 F1: 0.7471920356375976
    RidgeClassifierCV with HashingVectorizer. Precision: 0.773123404871145. Recall: 0.7508250825082509 F1: 0.7264662988071177
    
    AdaBoostClassifier with CountVectorizer. Precision: 0.5624601933877599. Recall: 0.641914191419142 F1: 0.5406036589115659
    AdaBoostClassifier with TfidfVectorizer. Precision: 0.5339725237583999. Recall: 0.6320132013201321 F1: 0.5699842045648011
    AdaBoostClassifier with HashingVectorizer. Precision: 0.53907862696382. Recall: 0.636963696369637 F1: 0.5593764668638859
    BaggingClassifier with CountVectorizer. Precision: 0.7669478999181969. Recall: 0.7557755775577558 F1: 0.7322443282888413
    BaggingClassifier with TfidfVectorizer. Precision: 0.7137424551617745. Recall: 0.7227722772277227 F1: 0.7128632696988642
    BaggingClassifier with HashingVectorizer. Precision: 0.7403858272509948. Recall: 0.7343234323432343 F1: 0.7135123764608311
    ExtraTreesClassifier with CountVectorizer. Precision: 0.7549552926895123. Recall: 0.7392739273927392 F1: 0.7160856739271999
    ExtraTreesClassifier with TfidfVectorizer. Precision: 0.7431510883398655. Recall: 0.731023102310231 F1: 0.7090809574438138
    ExtraTreesClassifier with HashingVectorizer. Precision: 0.7463822516886578. Recall: 0.7244224422442245 F1: 0.6877920606866322
    GradientBoostingClassifier with CountVectorizer. Precision: 0.7501302369042875. Recall: 0.7392739273927392 F1: 0.7154879043418036
    GradientBoostingClassifier with TfidfVectorizer. Precision: 0.7369080060565416. Recall: 0.7392739273927392 F1: 0.7272573566366475
    GradientBoostingClassifier with HashingVectorizer. Precision: 0.7634673319119184. Recall: 0.7524752475247525 F1: 0.7308409367026466
    DecisionTreeClassifier with CountVectorizer. Precision: 0.735328721362705. Recall: 0.735973597359736 F1: 0.7305161134303999
    DecisionTreeClassifier with TfidfVectorizer. Precision: 0.698245635113071. Recall: 0.698019801980198 F1: 0.6964044465362744
    DecisionTreeClassifier with HashingVectorizer. Precision: 0.7126253074833959. Recall: 0.7211221122112211 F1: 0.7117640914702404
    DummyClassifier with CountVectorizer. Precision: 0.4540685703626661. Recall: 0.46864686468646866 F1: 0.46081773101695783
    DummyClassifier with TfidfVectorizer. Precision: 0.49890161894036195. Recall: 0.5033003300330033 F1: 0.5007505170128534
    DummyClassifier with HashingVectorizer. Precision: 0.42700854957744705. Recall: 0.44224422442244227 F1: 0.4332406915254077
    PassiveAggressiveClassifier with CountVectorizer. Precision: 0.7465797901302191. Recall: 0.7508250825082509 F1: 0.7438667084320119
    PassiveAggressiveClassifier with TfidfVectorizer. Precision: 0.7467228883856221. Recall: 0.7508250825082509 F1: 0.7452169407498447
    PassiveAggressiveClassifier with HashingVectorizer. Precision: 0.7462649540850699. Recall: 0.7442244224422442 F1: 0.7294605227113802
    RidgeClassifier with CountVectorizer. Precision: 0.7508635193530426. Recall: 0.7524752475247525 F1: 0.7403578708823831
    RidgeClassifier with TfidfVectorizer. Precision: 0.7805865615724012. Recall: 0.7755775577557755 F1: 0.7649945842173317
    RidgeClassifier with HashingVectorizer. Precision: 0.7799049672409102. Recall: 0.7557755775577558 F1: 0.7304898640208674
    RidgeClassifierCV with CountVectorizer. Precision: 0.7569831829763922. Recall: 0.7425742574257426 F1: 0.7184048954830016
    RidgeClassifierCV with TfidfVectorizer. Precision: 0.7752205496206167. Recall: 0.764026402640264 F1: 0.7471920356375976
    RidgeClassifierCV with HashingVectorizer. Precision: 0.773123404871145. Recall: 0.7508250825082509 F1: 0.7264662988071177
    SGDClassifier with CountVectorizer. Precision: 0.727611968186066. Recall: 0.731023102310231 F1: 0.715472832375222
    SGDClassifier with TfidfVectorizer. Precision: 0.7557242160538156. Recall: 0.759075907590759 F1: 0.7512400362298152
    SGDClassifier with HashingVectorizer. Precision: 0.7691102634805675. Recall: 0.7607260726072608 F1: 0.7437883276145697
    
    AdaBoostClassifier with CountVectorizer. Precision: 0.5624601933877599. Recall: 0.641914191419142 F1: 0.5406036589115659
    AdaBoostClassifier with TfidfVectorizer. Precision: 0.5339725237583999. Recall: 0.6320132013201321 F1: 0.5699842045648011
    AdaBoostClassifier with HashingVectorizer. Precision: 0.53907862696382. Recall: 0.636963696369637 F1: 0.5593764668638859
    BaggingClassifier with CountVectorizer. Precision: 0.7669478999181969. Recall: 0.7557755775577558 F1: 0.7322443282888413
    BaggingClassifier with TfidfVectorizer. Precision: 0.7137424551617745. Recall: 0.7227722772277227 F1: 0.7128632696988642
    BaggingClassifier with HashingVectorizer. Precision: 0.7403858272509948. Recall: 0.7343234323432343 F1: 0.7135123764608311
    ExtraTreesClassifier with CountVectorizer. Precision: 0.7549552926895123. Recall: 0.7392739273927392 F1: 0.7160856739271999
    ExtraTreesClassifier with TfidfVectorizer. Precision: 0.7431510883398655. Recall: 0.731023102310231 F1: 0.7090809574438138
    ExtraTreesClassifier with HashingVectorizer. Precision: 0.7463822516886578. Recall: 0.7244224422442245 F1: 0.6877920606866322
    GradientBoostingClassifier with CountVectorizer. Precision: 0.7501302369042875. Recall: 0.7392739273927392 F1: 0.7154879043418036
    GradientBoostingClassifier with TfidfVectorizer. Precision: 0.7369080060565416. Recall: 0.7392739273927392 F1: 0.7272573566366475
    GradientBoostingClassifier with HashingVectorizer. Precision: 0.7634673319119184. Recall: 0.7524752475247525 F1: 0.7308409367026466
    DecisionTreeClassifier with CountVectorizer. Precision: 0.735328721362705. Recall: 0.735973597359736 F1: 0.7305161134303999
    DecisionTreeClassifier with TfidfVectorizer. Precision: 0.698245635113071. Recall: 0.698019801980198 F1: 0.6964044465362744
    DecisionTreeClassifier with HashingVectorizer. Precision: 0.7126253074833959. Recall: 0.7211221122112211 F1: 0.7117640914702404
    DummyClassifier with CountVectorizer. Precision: 0.4540685703626661. Recall: 0.46864686468646866 F1: 0.46081773101695783
    DummyClassifier with TfidfVectorizer. Precision: 0.49890161894036195. Recall: 0.5033003300330033 F1: 0.5007505170128534
    DummyClassifier with HashingVectorizer. Precision: 0.42700854957744705. Recall: 0.44224422442244227 F1: 0.4332406915254077
    PassiveAggressiveClassifier with CountVectorizer. Precision: 0.7465797901302191. Recall: 0.7508250825082509 F1: 0.7438667084320119
    PassiveAggressiveClassifier with TfidfVectorizer. Precision: 0.7467228883856221. Recall: 0.7508250825082509 F1: 0.7452169407498447
    PassiveAggressiveClassifier with HashingVectorizer. Precision: 0.7462649540850699. Recall: 0.7442244224422442 F1: 0.7294605227113802
    RidgeClassifier with CountVectorizer. Precision: 0.7508635193530426. Recall: 0.7524752475247525 F1: 0.7403578708823831
    RidgeClassifier with TfidfVectorizer. Precision: 0.7805865615724012. Recall: 0.7755775577557755 F1: 0.7649945842173317
    RidgeClassifier with HashingVectorizer. Precision: 0.7799049672409102. Recall: 0.7557755775577558 F1: 0.7304898640208674
    RidgeClassifierCV with CountVectorizer. Precision: 0.7569831829763922. Recall: 0.7425742574257426 F1: 0.7184048954830016
    RidgeClassifierCV with TfidfVectorizer. Precision: 0.7752205496206167. Recall: 0.764026402640264 F1: 0.7471920356375976
    RidgeClassifierCV with HashingVectorizer. Precision: 0.773123404871145. Recall: 0.7508250825082509 F1: 0.7264662988071177
    SGDClassifier with CountVectorizer. Precision: 0.727611968186066. Recall: 0.731023102310231 F1: 0.715472832375222
    SGDClassifier with TfidfVectorizer. Precision: 0.7557242160538156. Recall: 0.759075907590759 F1: 0.7512400362298152
    SGDClassifier with HashingVectorizer. Precision: 0.7691102634805675. Recall: 0.7607260726072608 F1: 0.7437883276145697
    OneVsRestClassifier with CountVectorizer. Precision: 0.7513248085942199. Recall: 0.7524752475247525 F1: 0.7433874270141145
    OneVsRestClassifier with TfidfVectorizer. Precision: 0.7791484178342143. Recall: 0.7755775577557755 F1: 0.7639174974446724
    OneVsRestClassifier with HashingVectorizer. Precision: 0.766372354164186. Recall: 0.7425742574257426 F1: 0.715775965092041
    
    AdaBoostClassifier with CountVectorizer. Precision: 0.5624601933877599. Recall: 0.641914191419142 F1: 0.5406036589115659
    AdaBoostClassifier with TfidfVectorizer. Precision: 0.5339725237583999. Recall: 0.6320132013201321 F1: 0.5699842045648011
    AdaBoostClassifier with HashingVectorizer. Precision: 0.53907862696382. Recall: 0.636963696369637 F1: 0.5593764668638859
    BaggingClassifier with CountVectorizer. Precision: 0.7669478999181969. Recall: 0.7557755775577558 F1: 0.7322443282888413
    BaggingClassifier with TfidfVectorizer. Precision: 0.7137424551617745. Recall: 0.7227722772277227 F1: 0.7128632696988642
    BaggingClassifier with HashingVectorizer. Precision: 0.7403858272509948. Recall: 0.7343234323432343 F1: 0.7135123764608311
    ExtraTreesClassifier with CountVectorizer. Precision: 0.7549552926895123. Recall: 0.7392739273927392 F1: 0.7160856739271999
    ExtraTreesClassifier with TfidfVectorizer. Precision: 0.7431510883398655. Recall: 0.731023102310231 F1: 0.7090809574438138
    ExtraTreesClassifier with HashingVectorizer. Precision: 0.7463822516886578. Recall: 0.7244224422442245 F1: 0.6877920606866322
    GradientBoostingClassifier with CountVectorizer. Precision: 0.7501302369042875. Recall: 0.7392739273927392 F1: 0.7154879043418036
    GradientBoostingClassifier with TfidfVectorizer. Precision: 0.7369080060565416. Recall: 0.7392739273927392 F1: 0.7272573566366475
    GradientBoostingClassifier with HashingVectorizer. Precision: 0.7634673319119184. Recall: 0.7524752475247525 F1: 0.7308409367026466
    DecisionTreeClassifier with CountVectorizer. Precision: 0.735328721362705. Recall: 0.735973597359736 F1: 0.7305161134303999
    DecisionTreeClassifier with TfidfVectorizer. Precision: 0.698245635113071. Recall: 0.698019801980198 F1: 0.6964044465362744
    DecisionTreeClassifier with HashingVectorizer. Precision: 0.7126253074833959. Recall: 0.7211221122112211 F1: 0.7117640914702404
    DummyClassifier with CountVectorizer. Precision: 0.4540685703626661. Recall: 0.46864686468646866 F1: 0.46081773101695783
    DummyClassifier with TfidfVectorizer. Precision: 0.49890161894036195. Recall: 0.5033003300330033 F1: 0.5007505170128534
    DummyClassifier with HashingVectorizer. Precision: 0.42700854957744705. Recall: 0.44224422442244227 F1: 0.4332406915254077
    PassiveAggressiveClassifier with CountVectorizer. Precision: 0.7465797901302191. Recall: 0.7508250825082509 F1: 0.7438667084320119
    PassiveAggressiveClassifier with TfidfVectorizer. Precision: 0.7467228883856221. Recall: 0.7508250825082509 F1: 0.7452169407498447
    PassiveAggressiveClassifier with HashingVectorizer. Precision: 0.7462649540850699. Recall: 0.7442244224422442 F1: 0.7294605227113802
    RidgeClassifier with CountVectorizer. Precision: 0.7508635193530426. Recall: 0.7524752475247525 F1: 0.7403578708823831
    RidgeClassifier with TfidfVectorizer. Precision: 0.7805865615724012. Recall: 0.7755775577557755 F1: 0.7649945842173317
    RidgeClassifier with HashingVectorizer. Precision: 0.7799049672409102. Recall: 0.7557755775577558 F1: 0.7304898640208674
    RidgeClassifierCV with CountVectorizer. Precision: 0.7569831829763922. Recall: 0.7425742574257426 F1: 0.7184048954830016
    RidgeClassifierCV with TfidfVectorizer. Precision: 0.7752205496206167. Recall: 0.764026402640264 F1: 0.7471920356375976
    RidgeClassifierCV with HashingVectorizer. Precision: 0.773123404871145. Recall: 0.7508250825082509 F1: 0.7264662988071177
    SGDClassifier with CountVectorizer. Precision: 0.727611968186066. Recall: 0.731023102310231 F1: 0.715472832375222
    SGDClassifier with TfidfVectorizer. Precision: 0.7557242160538156. Recall: 0.759075907590759 F1: 0.7512400362298152
    SGDClassifier with HashingVectorizer. Precision: 0.7691102634805675. Recall: 0.7607260726072608 F1: 0.7437883276145697
    OneVsRestClassifier with CountVectorizer. Precision: 0.7513248085942199. Recall: 0.7524752475247525 F1: 0.7433874270141145
    OneVsRestClassifier with TfidfVectorizer. Precision: 0.7791484178342143. Recall: 0.7755775577557755 F1: 0.7639174974446724
    OneVsRestClassifier with HashingVectorizer. Precision: 0.766372354164186. Recall: 0.7425742574257426 F1: 0.715775965092041
    OneVsRestClassifier with CountVectorizer. Precision: 0.7716626997276231. Recall: 0.764026402640264 F1: 0.7512970995819267
    OneVsRestClassifier with TfidfVectorizer. Precision: 0.7736418588001085. Recall: 0.7326732673267327 F1: 0.6904061700521771
    OneVsRestClassifier with HashingVectorizer. Precision: 0.757304553392965. Recall: 0.7178217821782178 F1: 0.6702866723581539
    
    AdaBoostClassifier with CountVectorizer. Precision: 0.5624601933877599. Recall: 0.641914191419142 F1: 0.5406036589115659
    AdaBoostClassifier with TfidfVectorizer. Precision: 0.5339725237583999. Recall: 0.6320132013201321 F1: 0.5699842045648011
    AdaBoostClassifier with HashingVectorizer. Precision: 0.53907862696382. Recall: 0.636963696369637 F1: 0.5593764668638859
    BaggingClassifier with CountVectorizer. Precision: 0.7669478999181969. Recall: 0.7557755775577558 F1: 0.7322443282888413
    BaggingClassifier with TfidfVectorizer. Precision: 0.7137424551617745. Recall: 0.7227722772277227 F1: 0.7128632696988642
    BaggingClassifier with HashingVectorizer. Precision: 0.7403858272509948. Recall: 0.7343234323432343 F1: 0.7135123764608311
    ExtraTreesClassifier with CountVectorizer. Precision: 0.7549552926895123. Recall: 0.7392739273927392 F1: 0.7160856739271999
    ExtraTreesClassifier with TfidfVectorizer. Precision: 0.7431510883398655. Recall: 0.731023102310231 F1: 0.7090809574438138
    ExtraTreesClassifier with HashingVectorizer. Precision: 0.7463822516886578. Recall: 0.7244224422442245 F1: 0.6877920606866322
    GradientBoostingClassifier with CountVectorizer. Precision: 0.7501302369042875. Recall: 0.7392739273927392 F1: 0.7154879043418036
    GradientBoostingClassifier with TfidfVectorizer. Precision: 0.7369080060565416. Recall: 0.7392739273927392 F1: 0.7272573566366475
    GradientBoostingClassifier with HashingVectorizer. Precision: 0.7634673319119184. Recall: 0.7524752475247525 F1: 0.7308409367026466
    DecisionTreeClassifier with CountVectorizer. Precision: 0.735328721362705. Recall: 0.735973597359736 F1: 0.7305161134303999
    DecisionTreeClassifier with TfidfVectorizer. Precision: 0.698245635113071. Recall: 0.698019801980198 F1: 0.6964044465362744
    DecisionTreeClassifier with HashingVectorizer. Precision: 0.7126253074833959. Recall: 0.7211221122112211 F1: 0.7117640914702404
    DummyClassifier with CountVectorizer. Precision: 0.4540685703626661. Recall: 0.46864686468646866 F1: 0.46081773101695783
    DummyClassifier with TfidfVectorizer. Precision: 0.49890161894036195. Recall: 0.5033003300330033 F1: 0.5007505170128534
    DummyClassifier with HashingVectorizer. Precision: 0.42700854957744705. Recall: 0.44224422442244227 F1: 0.4332406915254077
    PassiveAggressiveClassifier with CountVectorizer. Precision: 0.7465797901302191. Recall: 0.7508250825082509 F1: 0.7438667084320119
    PassiveAggressiveClassifier with TfidfVectorizer. Precision: 0.7467228883856221. Recall: 0.7508250825082509 F1: 0.7452169407498447
    PassiveAggressiveClassifier with HashingVectorizer. Precision: 0.7462649540850699. Recall: 0.7442244224422442 F1: 0.7294605227113802
    RidgeClassifier with CountVectorizer. Precision: 0.7508635193530426. Recall: 0.7524752475247525 F1: 0.7403578708823831
    RidgeClassifier with TfidfVectorizer. Precision: 0.7805865615724012. Recall: 0.7755775577557755 F1: 0.7649945842173317
    RidgeClassifier with HashingVectorizer. Precision: 0.7799049672409102. Recall: 0.7557755775577558 F1: 0.7304898640208674
    RidgeClassifierCV with CountVectorizer. Precision: 0.7569831829763922. Recall: 0.7425742574257426 F1: 0.7184048954830016
    RidgeClassifierCV with TfidfVectorizer. Precision: 0.7752205496206167. Recall: 0.764026402640264 F1: 0.7471920356375976
    RidgeClassifierCV with HashingVectorizer. Precision: 0.773123404871145. Recall: 0.7508250825082509 F1: 0.7264662988071177
    SGDClassifier with CountVectorizer. Precision: 0.727611968186066. Recall: 0.731023102310231 F1: 0.715472832375222
    SGDClassifier with TfidfVectorizer. Precision: 0.7557242160538156. Recall: 0.759075907590759 F1: 0.7512400362298152
    SGDClassifier with HashingVectorizer. Precision: 0.7691102634805675. Recall: 0.7607260726072608 F1: 0.7437883276145697
    OneVsRestClassifier with CountVectorizer. Precision: 0.7513248085942199. Recall: 0.7524752475247525 F1: 0.7433874270141145
    OneVsRestClassifier with TfidfVectorizer. Precision: 0.7791484178342143. Recall: 0.7755775577557755 F1: 0.7639174974446724
    OneVsRestClassifier with HashingVectorizer. Precision: 0.766372354164186. Recall: 0.7425742574257426 F1: 0.715775965092041
    OneVsRestClassifier with CountVectorizer. Precision: 0.7716626997276231. Recall: 0.764026402640264 F1: 0.7512970995819267
    OneVsRestClassifier with TfidfVectorizer. Precision: 0.7736418588001085. Recall: 0.7326732673267327 F1: 0.6904061700521771
    OneVsRestClassifier with HashingVectorizer. Precision: 0.757304553392965. Recall: 0.7178217821782178 F1: 0.6702866723581539
    KNeighborsClassifier with CountVectorizer. Precision: 0.5943769408367612. Recall: 0.6072607260726073 F1: 0.5943065854702626
    KNeighborsClassifier with TfidfVectorizer. Precision: 0.7364647195250305. Recall: 0.7376237623762376 F1: 0.7351590548056839
    KNeighborsClassifier with HashingVectorizer. Precision: 0.6839960953931562. Recall: 0.6897689768976898 F1: 0.6839337734622661
    
    AdaBoostClassifier with CountVectorizer. Precision: 0.5624601933877599. Recall: 0.641914191419142 F1: 0.5406036589115659
    AdaBoostClassifier with TfidfVectorizer. Precision: 0.5339725237583999. Recall: 0.6320132013201321 F1: 0.5699842045648011
    AdaBoostClassifier with HashingVectorizer. Precision: 0.53907862696382. Recall: 0.636963696369637 F1: 0.5593764668638859
    BaggingClassifier with CountVectorizer. Precision: 0.7669478999181969. Recall: 0.7557755775577558 F1: 0.7322443282888413
    BaggingClassifier with TfidfVectorizer. Precision: 0.7137424551617745. Recall: 0.7227722772277227 F1: 0.7128632696988642
    BaggingClassifier with HashingVectorizer. Precision: 0.7403858272509948. Recall: 0.7343234323432343 F1: 0.7135123764608311
    ExtraTreesClassifier with CountVectorizer. Precision: 0.7549552926895123. Recall: 0.7392739273927392 F1: 0.7160856739271999
    ExtraTreesClassifier with TfidfVectorizer. Precision: 0.7431510883398655. Recall: 0.731023102310231 F1: 0.7090809574438138
    ExtraTreesClassifier with HashingVectorizer. Precision: 0.7463822516886578. Recall: 0.7244224422442245 F1: 0.6877920606866322
    GradientBoostingClassifier with CountVectorizer. Precision: 0.7501302369042875. Recall: 0.7392739273927392 F1: 0.7154879043418036
    GradientBoostingClassifier with TfidfVectorizer. Precision: 0.7369080060565416. Recall: 0.7392739273927392 F1: 0.7272573566366475
    GradientBoostingClassifier with HashingVectorizer. Precision: 0.7634673319119184. Recall: 0.7524752475247525 F1: 0.7308409367026466
    DecisionTreeClassifier with CountVectorizer. Precision: 0.735328721362705. Recall: 0.735973597359736 F1: 0.7305161134303999
    DecisionTreeClassifier with TfidfVectorizer. Precision: 0.698245635113071. Recall: 0.698019801980198 F1: 0.6964044465362744
    DecisionTreeClassifier with HashingVectorizer. Precision: 0.7126253074833959. Recall: 0.7211221122112211 F1: 0.7117640914702404
    DummyClassifier with CountVectorizer. Precision: 0.4540685703626661. Recall: 0.46864686468646866 F1: 0.46081773101695783
    DummyClassifier with TfidfVectorizer. Precision: 0.49890161894036195. Recall: 0.5033003300330033 F1: 0.5007505170128534
    DummyClassifier with HashingVectorizer. Precision: 0.42700854957744705. Recall: 0.44224422442244227 F1: 0.4332406915254077
    PassiveAggressiveClassifier with CountVectorizer. Precision: 0.7465797901302191. Recall: 0.7508250825082509 F1: 0.7438667084320119
    PassiveAggressiveClassifier with TfidfVectorizer. Precision: 0.7467228883856221. Recall: 0.7508250825082509 F1: 0.7452169407498447
    PassiveAggressiveClassifier with HashingVectorizer. Precision: 0.7462649540850699. Recall: 0.7442244224422442 F1: 0.7294605227113802
    RidgeClassifier with CountVectorizer. Precision: 0.7508635193530426. Recall: 0.7524752475247525 F1: 0.7403578708823831
    RidgeClassifier with TfidfVectorizer. Precision: 0.7805865615724012. Recall: 0.7755775577557755 F1: 0.7649945842173317
    RidgeClassifier with HashingVectorizer. Precision: 0.7799049672409102. Recall: 0.7557755775577558 F1: 0.7304898640208674
    RidgeClassifierCV with CountVectorizer. Precision: 0.7569831829763922. Recall: 0.7425742574257426 F1: 0.7184048954830016
    RidgeClassifierCV with TfidfVectorizer. Precision: 0.7752205496206167. Recall: 0.764026402640264 F1: 0.7471920356375976
    RidgeClassifierCV with HashingVectorizer. Precision: 0.773123404871145. Recall: 0.7508250825082509 F1: 0.7264662988071177
    SGDClassifier with CountVectorizer. Precision: 0.727611968186066. Recall: 0.731023102310231 F1: 0.715472832375222
    SGDClassifier with TfidfVectorizer. Precision: 0.7557242160538156. Recall: 0.759075907590759 F1: 0.7512400362298152
    SGDClassifier with HashingVectorizer. Precision: 0.7691102634805675. Recall: 0.7607260726072608 F1: 0.7437883276145697
    OneVsRestClassifier with CountVectorizer. Precision: 0.7513248085942199. Recall: 0.7524752475247525 F1: 0.7433874270141145
    OneVsRestClassifier with TfidfVectorizer. Precision: 0.7791484178342143. Recall: 0.7755775577557755 F1: 0.7639174974446724
    OneVsRestClassifier with HashingVectorizer. Precision: 0.766372354164186. Recall: 0.7425742574257426 F1: 0.715775965092041
    OneVsRestClassifier with CountVectorizer. Precision: 0.7716626997276231. Recall: 0.764026402640264 F1: 0.7512970995819267
    OneVsRestClassifier with TfidfVectorizer. Precision: 0.7736418588001085. Recall: 0.7326732673267327 F1: 0.6904061700521771
    OneVsRestClassifier with HashingVectorizer. Precision: 0.757304553392965. Recall: 0.7178217821782178 F1: 0.6702866723581539
    KNeighborsClassifier with CountVectorizer. Precision: 0.5943769408367612. Recall: 0.6072607260726073 F1: 0.5943065854702626
    KNeighborsClassifier with TfidfVectorizer. Precision: 0.7364647195250305. Recall: 0.7376237623762376 F1: 0.7351590548056839
    KNeighborsClassifier with HashingVectorizer. Precision: 0.6839960953931562. Recall: 0.6897689768976898 F1: 0.6839337734622661
    BernoulliNB with CountVectorizer. Precision: 0.606308382605045. Recall: 0.6584158415841584 F1: 0.566162386902002
    BernoulliNB with TfidfVectorizer. Precision: 0.606308382605045. Recall: 0.6584158415841584 F1: 0.566162386902002
    BernoulliNB with HashingVectorizer. Precision: 0.37885446960537633. Recall: 0.6155115511551155 F1: 0.4690210594093116
    
    AdaBoostClassifier with CountVectorizer. Precision: 0.5624601933877599. Recall: 0.641914191419142 F1: 0.5406036589115659
    AdaBoostClassifier with TfidfVectorizer. Precision: 0.5339725237583999. Recall: 0.6320132013201321 F1: 0.5699842045648011
    AdaBoostClassifier with HashingVectorizer. Precision: 0.53907862696382. Recall: 0.636963696369637 F1: 0.5593764668638859
    BaggingClassifier with CountVectorizer. Precision: 0.7669478999181969. Recall: 0.7557755775577558 F1: 0.7322443282888413
    BaggingClassifier with TfidfVectorizer. Precision: 0.7137424551617745. Recall: 0.7227722772277227 F1: 0.7128632696988642
    BaggingClassifier with HashingVectorizer. Precision: 0.7403858272509948. Recall: 0.7343234323432343 F1: 0.7135123764608311
    ExtraTreesClassifier with CountVectorizer. Precision: 0.7549552926895123. Recall: 0.7392739273927392 F1: 0.7160856739271999
    ExtraTreesClassifier with TfidfVectorizer. Precision: 0.7431510883398655. Recall: 0.731023102310231 F1: 0.7090809574438138
    ExtraTreesClassifier with HashingVectorizer. Precision: 0.7463822516886578. Recall: 0.7244224422442245 F1: 0.6877920606866322
    GradientBoostingClassifier with CountVectorizer. Precision: 0.7501302369042875. Recall: 0.7392739273927392 F1: 0.7154879043418036
    GradientBoostingClassifier with TfidfVectorizer. Precision: 0.7369080060565416. Recall: 0.7392739273927392 F1: 0.7272573566366475
    GradientBoostingClassifier with HashingVectorizer. Precision: 0.7634673319119184. Recall: 0.7524752475247525 F1: 0.7308409367026466
    DecisionTreeClassifier with CountVectorizer. Precision: 0.735328721362705. Recall: 0.735973597359736 F1: 0.7305161134303999
    DecisionTreeClassifier with TfidfVectorizer. Precision: 0.698245635113071. Recall: 0.698019801980198 F1: 0.6964044465362744
    DecisionTreeClassifier with HashingVectorizer. Precision: 0.7126253074833959. Recall: 0.7211221122112211 F1: 0.7117640914702404
    DummyClassifier with CountVectorizer. Precision: 0.4540685703626661. Recall: 0.46864686468646866 F1: 0.46081773101695783
    DummyClassifier with TfidfVectorizer. Precision: 0.49890161894036195. Recall: 0.5033003300330033 F1: 0.5007505170128534
    DummyClassifier with HashingVectorizer. Precision: 0.42700854957744705. Recall: 0.44224422442244227 F1: 0.4332406915254077
    PassiveAggressiveClassifier with CountVectorizer. Precision: 0.7465797901302191. Recall: 0.7508250825082509 F1: 0.7438667084320119
    PassiveAggressiveClassifier with TfidfVectorizer. Precision: 0.7467228883856221. Recall: 0.7508250825082509 F1: 0.7452169407498447
    PassiveAggressiveClassifier with HashingVectorizer. Precision: 0.7462649540850699. Recall: 0.7442244224422442 F1: 0.7294605227113802
    RidgeClassifier with CountVectorizer. Precision: 0.7508635193530426. Recall: 0.7524752475247525 F1: 0.7403578708823831
    RidgeClassifier with TfidfVectorizer. Precision: 0.7805865615724012. Recall: 0.7755775577557755 F1: 0.7649945842173317
    RidgeClassifier with HashingVectorizer. Precision: 0.7799049672409102. Recall: 0.7557755775577558 F1: 0.7304898640208674
    RidgeClassifierCV with CountVectorizer. Precision: 0.7569831829763922. Recall: 0.7425742574257426 F1: 0.7184048954830016
    RidgeClassifierCV with TfidfVectorizer. Precision: 0.7752205496206167. Recall: 0.764026402640264 F1: 0.7471920356375976
    RidgeClassifierCV with HashingVectorizer. Precision: 0.773123404871145. Recall: 0.7508250825082509 F1: 0.7264662988071177
    SGDClassifier with CountVectorizer. Precision: 0.727611968186066. Recall: 0.731023102310231 F1: 0.715472832375222
    SGDClassifier with TfidfVectorizer. Precision: 0.7557242160538156. Recall: 0.759075907590759 F1: 0.7512400362298152
    SGDClassifier with HashingVectorizer. Precision: 0.7691102634805675. Recall: 0.7607260726072608 F1: 0.7437883276145697
    OneVsRestClassifier with CountVectorizer. Precision: 0.7513248085942199. Recall: 0.7524752475247525 F1: 0.7433874270141145
    OneVsRestClassifier with TfidfVectorizer. Precision: 0.7791484178342143. Recall: 0.7755775577557755 F1: 0.7639174974446724
    OneVsRestClassifier with HashingVectorizer. Precision: 0.766372354164186. Recall: 0.7425742574257426 F1: 0.715775965092041
    OneVsRestClassifier with CountVectorizer. Precision: 0.7716626997276231. Recall: 0.764026402640264 F1: 0.7512970995819267
    OneVsRestClassifier with TfidfVectorizer. Precision: 0.7736418588001085. Recall: 0.7326732673267327 F1: 0.6904061700521771
    OneVsRestClassifier with HashingVectorizer. Precision: 0.757304553392965. Recall: 0.7178217821782178 F1: 0.6702866723581539
    KNeighborsClassifier with CountVectorizer. Precision: 0.5943769408367612. Recall: 0.6072607260726073 F1: 0.5943065854702626
    KNeighborsClassifier with TfidfVectorizer. Precision: 0.7364647195250305. Recall: 0.7376237623762376 F1: 0.7351590548056839
    KNeighborsClassifier with HashingVectorizer. Precision: 0.6839960953931562. Recall: 0.6897689768976898 F1: 0.6839337734622661
    BernoulliNB with CountVectorizer. Precision: 0.606308382605045. Recall: 0.6584158415841584 F1: 0.566162386902002
    BernoulliNB with TfidfVectorizer. Precision: 0.606308382605045. Recall: 0.6584158415841584 F1: 0.566162386902002
    BernoulliNB with HashingVectorizer. Precision: 0.37885446960537633. Recall: 0.6155115511551155 F1: 0.4690210594093116
    RandomForestClassifier with CountVectorizer. Precision: 0.769245961863267. Recall: 0.7392739273927392 F1: 0.7020001585522605
    RandomForestClassifier with TfidfVectorizer. Precision: 0.7516099926097494. Recall: 0.735973597359736 F1: 0.7002173489238325
    RandomForestClassifier with HashingVectorizer. Precision: 0.7511624407692628. Recall: 0.7095709570957096 F1: 0.6575768526699561
    



As we can see, each model was performed with three different Vectorizer algorithms and outputed your related metrics scores.


## Conclusions
