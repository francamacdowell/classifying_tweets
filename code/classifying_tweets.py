# Importing needed libs:

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
import warnings

# Ignoring warnings outputs:
warnings.filterwarnings("ignore")

# Function to perform a list of models with a list of Vectorizers:
def perform_models(classifiers, vectorizers, X_train, X_test, y_train, y_test):
    string = ''
    # For each model classifier for each Vectorizer Algorithm: 
    for classifier in classifiers:
        for vectorizer in vectorizers:
            string += classifier.__class__.__name__ + ' with ' + vectorizer.__class__.__name__

            # Train
            vectorize_text_train = vectorizer.fit_transform(X_train)
            classifier.fit(vectorize_text_train, y_train)
            # Evaluating model
            vectorize_text_test = vectorizer.transform(X_test)

            predicteds = classifier.predict(vectorize_text_test)
            # Getting score metrics
            precision, recall, f1, supp = precision_recall_fscore_support(y_test, predicteds,
                                                                          average='weighted')
            string += '. Has precision: ' + str(precision) + '. recall: ' + str(recall) + ' F1: ' + str(f1) + '\n'
        print(string + '\n')
        string = ''

# Reading dataset:
data = pd.read_csv('../tweets.csv')
# Dropping useless column:
data = data.drop(columns='id')

# Splitting randomly our dataset into train/test instances:
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['type'], test_size=0.3, shuffle=True)
## Here we could use Cross-validation too -> sklearn.model_selection.KFold

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
