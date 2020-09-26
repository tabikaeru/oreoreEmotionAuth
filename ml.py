from sklearn.model_selection import train_test_split
import csv
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model


def train():
    with open('data.csv', 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

    texts = []
    for i, point in enumerate(data):
        texts.append(data[i][3])
    texts.pop(0)

    labels = []
    for i, point in enumerate(data):
        labels.append(data[i][1])
    labels.pop(0)

    text_train, text_test, label_train, label_test = train_test_split(
        texts, labels, test_size=0.1)

    vectorizer = TfidfVectorizer()
    train_matrix = vectorizer.fit_transform(text_train)
    joblib.dump(vectorizer, './classfier/tfIDF.pkl.cmp', compress=True)
    test_matrix = vectorizer.transform(text_test)

    clf1 = MultinomialNB()
    clf1.fit(train_matrix, label_train)
    joblib.dump(clf1, './classfier/clf1.pkl.cmp', compress=True)

    print(clf1.score(train_matrix, label_train))
    print(clf1.score(test_matrix, label_test))

    clf2 = RandomForestClassifier(n_estimators=100)

    clf2.fit(train_matrix, label_train)
    joblib.dump(clf2, './classfier/clf2.pkl.cmp', compress=True)

    print(clf2.score(train_matrix, label_train))
    print(clf2.score(test_matrix, label_test))

    clf3 = linear_model.SGDClassifier(loss="hinge")
    clf3.fit(train_matrix, label_train)
    joblib.dump(clf3, './classfier/clf3.pkl.cmp', compress=True)

    print(clf3.score(train_matrix, label_train))
    print(clf3.score(test_matrix, label_test))


def classifier(num):
    vectorizer = joblib.load('./classfier/tfIDF.pkl.cmp')
    if(num == 2):
        clf2 = joblib.load('./classfier/clf2.pkl.cmp')
        return vectorizer, clf2

    if(num == 2):
        clf2 = joblib.load('./classfier/clf2.pkl.cmp')
        return vectorizer, clf2
    if(num == 2):
        clf2 = joblib.load('./classfier/clf2.pkl.cmp')
        return vectorizer, clf2


def predict(text, vectorizer, clf):
    test_matrixed = vectorizer.transform([text])
    res = clf.predict(test_matrixed)[0]
    return res


if __name__ == "__main__":
    # train()
    test_set = [
        "I like you",
        "I hate you",
        "I suspect you",
    ]

    vectorizer, clf = classifier(2)
    for text in test_set:
        res = predict(text, vectorizer, clf)
        print(res)
