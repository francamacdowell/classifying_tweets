from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import *
from sklearn.dummy import *
from sklearn.ensemble import *
from sklearn.neighbors import *
from sklearn.tree import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import *
from sklearn.multiclass import *
from sklearn.svm import *
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd


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
            string += '. Has precision: ' + str(precision) + '. recall: ' + str(recall) + ' F1: ' + str(f1) + '\n'
            print(string)


data = pd.read_csv('tweets.csv')
data = data.drop(columns='id')

X_train, X_test, y_train, y_test = train_test_split(data['text'], data['type'], test_size=0.3, shuffle=True)
#Here we could use Cross-validation too -> sklearn.model_selection.KFold

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
