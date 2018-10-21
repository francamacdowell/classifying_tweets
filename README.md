
# Supervised Tweets Prediction

## Introduction

   This project is a small part of a bigger project from my University where the objective was to create an application (mobile and web), which would help people to combat mosquitos that are vectors of _Zika_ virus, reporting their focus (propitious places to reproduce).
   To make the platform more precisely, we collected data from Twitter (tweets through _keywords_) that could help us to point suspicous places in the map and classified the labels manually.

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

This project consists in build a model to predict a tweet.

The labels are:

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
        print(string + '\n')
        string = ''

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
    
    
    BaggingClassifier with CountVectorizer. Precision: 0.7592065792469417. Recall: 0.7425742574257426 F1: 0.7175685252875926
    BaggingClassifier with TfidfVectorizer. Precision: 0.7017153480332634. Recall: 0.7112211221122112 F1: 0.7002951368809374
    BaggingClassifier with HashingVectorizer. Precision: 0.7398663343519417. Recall: 0.735973597359736 F1: 0.7153822033103645
    
    
    ExtraTreesClassifier with CountVectorizer. Precision: 0.7708943309092993. Recall: 0.7541254125412541 F1: 0.7312140616229972
    ExtraTreesClassifier with TfidfVectorizer. Precision: 0.7208515639089595. Recall: 0.7178217821782178 F1: 0.6840368561511873
    ExtraTreesClassifier with HashingVectorizer. Precision: 0.7188958038661009. Recall: 0.7013201320132013 F1: 0.6634216834471887
    
    
    GradientBoostingClassifier with CountVectorizer. Precision: 0.7571338056047919. Recall: 0.7442244224422442 F1: 0.7194351117382956
    GradientBoostingClassifier with TfidfVectorizer. Precision: 0.7345375083615191. Recall: 0.7376237623762376 F1: 0.7254936944747302
    GradientBoostingClassifier with HashingVectorizer. Precision: 0.7618884546888985. Recall: 0.7541254125412541 F1: 0.7324379967438627
    
    
    DecisionTreeClassifier with CountVectorizer. Precision: 0.7257489839006197. Recall: 0.731023102310231 F1: 0.7239472058586308
    DecisionTreeClassifier with TfidfVectorizer. Precision: 0.692296925821411. Recall: 0.698019801980198 F1: 0.6926928197382971
    DecisionTreeClassifier with HashingVectorizer. Precision: 0.704620651447182. Recall: 0.7128712871287128 F1: 0.7047927851595592
    
    
    DummyClassifier with CountVectorizer. Precision: 0.43676643271324395. Recall: 0.47194719471947194 F1: 0.45260021179935955
    DummyClassifier with TfidfVectorizer. Precision: 0.450128539234402. Recall: 0.4801980198019802 F1: 0.4624960843158467
    DummyClassifier with HashingVectorizer. Precision: 0.4378578033933713. Recall: 0.45874587458745875 F1: 0.44675904179344766
    
    
    PassiveAggressiveClassifier with CountVectorizer. Precision: 0.7456301461411649. Recall: 0.7508250825082509 F1: 0.745674942648506
    PassiveAggressiveClassifier with TfidfVectorizer. Precision: 0.7574595027205909. Recall: 0.7607260726072608 F1: 0.7539151110113095
    PassiveAggressiveClassifier with HashingVectorizer. Precision: 0.7525242646449709. Recall: 0.7541254125412541 F1: 0.745436858580237
    
    
    RidgeClassifier with CountVectorizer. Precision: 0.7508635193530426. Recall: 0.7524752475247525 F1: 0.7403578708823831
    RidgeClassifier with TfidfVectorizer. Precision: 0.7805865615724012. Recall: 0.7755775577557755 F1: 0.7649945842173317
    RidgeClassifier with HashingVectorizer. Precision: 0.7799049672409102. Recall: 0.7557755775577558 F1: 0.7304898640208674
    
    
    RidgeClassifierCV with CountVectorizer. Precision: 0.7569831829763922. Recall: 0.7425742574257426 F1: 0.7184048954830016
    RidgeClassifierCV with TfidfVectorizer. Precision: 0.7752205496206167. Recall: 0.764026402640264 F1: 0.7471920356375976
    RidgeClassifierCV with HashingVectorizer. Precision: 0.773123404871145. Recall: 0.7508250825082509 F1: 0.7264662988071177
    
    
    SGDClassifier with CountVectorizer. Precision: 0.7224664754287345. Recall: 0.7227722772277227 F1: 0.7203614497294109
    SGDClassifier with TfidfVectorizer. Precision: 0.7709886838957954. Recall: 0.7722772277227723 F1: 0.7674747421954966
    SGDClassifier with HashingVectorizer. Precision: 0.7647981875014728. Recall: 0.7656765676567657 F1: 0.7583988652692049
    
    
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
    
    
    RandomForestClassifier with CountVectorizer. Precision: 0.7797398502844047. Recall: 0.7508250825082509 F1: 0.7186157335483743
    RandomForestClassifier with TfidfVectorizer. Precision: 0.7543498853197332. Recall: 0.731023102310231 F1: 0.6930459998013869
    RandomForestClassifier with HashingVectorizer. Precision: 0.7701520394757921. Recall: 0.7244224422442245 F1: 0.6755993221387744
    
    



As we can see, each model was performed with three different Vectorizer algorithms and outputed your related metrics scores.


## Conclusions
