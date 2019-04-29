#!/bin/python

mode = 't'

from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer

tokenizer = RegexpTokenizer(r'\w+')
stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')
stemmer = PorterStemmer()

#if mode == 's':
#    stop_words = set(dict.fromkeys([stemmer.stem(word) for word in stop_words]))

def tokenize(text):
    tokens = tokenizer.tokenize(text)
    #stems = [stemmer.stem(item) for item in tokens]

    #res = tokens if mode == 't' else stems
    #for w in stop_words:
    #    while w in res:
    #        res.remove(w)
    #print(res)
    return tokens

def read_files(tarfname):
    """Read the training and development data from the sentiment tar file.
    The returned object contains various fields that store sentiment data, such as:

    train_data,dev_data: array of documents (array of words)
    train_fnames,dev_fnames: list of filenames of the doccuments (same length as data)
    train_labels,dev_labels: the true string label for each document (same length as data)

    The data is also preprocessed for use with scikit-learn, as:

    count_vec: CountVectorizer used to process the data (for reapplication on new data)
    trainX,devX: array of vectors representing Bags of Words, i.e. documents processed through the vectorizer
    le: LabelEncoder, i.e. a mapper from string labels to ints (stored for reapplication)
    target_labels: List of labels (same order as used in le)
    trainy,devy: array of int labels, one for each document
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    trainname = "train.tsv"
    devname = "dev.tsv"
    for member in tar.getmembers():
        if 'train.tsv' in member.name:
            trainname = member.name
        elif 'dev.tsv' in member.name:
            devname = member.name
            
            
    #1. Can add a list for positive or negative words, such as good/bad. When encounter them, add 3 more times
    class Data: pass
    sentiment = Data()
    print("-- train data")
    sentiment.train_data, sentiment.train_labels = read_tsv(tar,trainname)
    #print(sentiment.train_data[1])
    #print([stemmer.stem(item) for item in tokenizer.tokenize(sentiment.train_data[1])])
    print(len(sentiment.train_data))
    print("-- dev data")
    sentiment.dev_data, sentiment.dev_labels = read_tsv(tar, devname)
    print(len(sentiment.dev_data))
    print("-- transforming data and labels")
    from sklearn.feature_extraction.text import TfidfVectorizer
    #sentiment.count_vect = TfidfVectorizer(tokenizer=tokenize,stop_words='english',sublinear_tf=True,smooth_idf=False,use_idf=True,binary=True)
    sentiment.count_vect = TfidfVectorizer(tokenizer=tokenize,sublinear_tf=False,smooth_idf=False,use_idf=True,binary=True,max_df = 3000)
    sentiment.trainX = sentiment.count_vect.fit_transform(sentiment.train_data)
    sentiment.devX = sentiment.count_vect.transform(sentiment.dev_data)
    print("train shape: ", sentiment.trainX.shape)
    print("dev shape: ",sentiment.devX.shape)
    from sklearn import preprocessing
    sentiment.le = preprocessing.LabelEncoder()
    sentiment.le.fit(sentiment.train_labels)
    sentiment.target_labels = sentiment.le.classes_
    sentiment.trainy = sentiment.le.transform(sentiment.train_labels)
    sentiment.devy = sentiment.le.transform(sentiment.dev_labels)
    tar.close()
    return sentiment

def read_unlabeled(tarfname, sentiment):
    """Reads the unlabeled data.

    The returned object contains three fields that represent the unlabeled data.

    data: documents, represented as sequence of words
    fnames: list of filenames, one for each document
    X: bag of word vector for each document, using the sentiment.vectorizer
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    class Data: pass
    unlabeled = Data()
    unlabeled.data = []
    
    unlabeledname = "unlabeled.tsv"
    for member in tar.getmembers():
        if 'unlabeled.tsv' in member.name:
            unlabeledname = member.name
            
    print(unlabeledname)
    tf = tar.extractfile(unlabeledname)
    for line in tf:
        line = line.decode("utf-8")
        text = line.strip()
        unlabeled.data.append(text)
        
            
    unlabeled.X = sentiment.count_vect.transform(unlabeled.data)
    print(unlabeled.X.shape)
    tar.close()
    return unlabeled

def read_tsv(tar, fname):
    member = tar.getmember(fname)
    print(member.name)
    tf = tar.extractfile(member)
    data = []
    labels = []
    for line in tf:
        line = line.decode("utf-8")
        (label,text) = line.strip().split("\t")
        labels.append(label)
        data.append(text)
    return data, labels

def write_pred_kaggle_file(unlabeled, cls, outfname, sentiment):
    """Writes the predictions in Kaggle format.

    Given the unlabeled object, classifier, outputfilename, and the sentiment object,
    this function write sthe predictions of the classifier on the unlabeled data and
    writes it to the outputfilename. The sentiment object is required to ensure
    consistent label names.
    """
    yp = cls.predict(unlabeled.X)
    labels = sentiment.le.inverse_transform(yp)
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    for i in range(len(unlabeled.data)):
        f.write(str(i+1))
        f.write(",")
        f.write(labels[i])
        f.write("\n")
    f.close()


def write_gold_kaggle_file(tsvfile, outfname):
    """Writes the output Kaggle file of the truth.

    You will not be able to run this code, since the tsvfile is not
    accessible to you (it is the test labels).
    """
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    i = 0
    with open(tsvfile, 'r') as tf:
        for line in tf:
            (label,review) = line.strip().split("\t")
            i += 1
            f.write(str(i))
            f.write(",")
            f.write(label)
            f.write("\n")
    f.close()

def write_basic_kaggle_file(tsvfile, outfname):
    """Writes the output Kaggle file of the naive baseline.

    This baseline predicts POSITIVE for all the instances.
    """
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    i = 0
    with open(tsvfile, 'r') as tf:
        for line in tf:
            (label,review) = line.strip().split("\t")
            i += 1
            f.write(str(i))
            f.write(",")
            f.write("POSITIVE")
            f.write("\n")
    f.close()

def semi_train(cls):
    print("\n")
    print("-"*100)
    print("\nReading unlabeled data")
    unlabeled = read_unlabeled(tarfname, sentiment)
    #print("len: ", len(unlabeled.data))
    print("\nTraining semi-supervised classifier")
    #print("x: ",unlabeled.X)
    #tok = []
    #pred = []
    count = 0

    import numpy as np
    from scipy.sparse import vstack
    percent = int(len(unlabeled.data))
    print("percent: ",percent)

    print(sentiment.trainy)
    used = dict()
    conf = 0.8
    uLen = len(unlabeled.data)
    stop_dis = 1000
    prev = 0
    while True:
        found = False
        confident_labels = cls.predict_proba(unlabeled.X[:int((uLen/10)),:])
        for index,p in enumerate(confident_labels):
            y = -1
            if index in used:
                continue
            if p[0] >= conf or p[1] >= conf:
                y = cls.predict(unlabeled.X[index])
                used[index] = y[0]
            if y != -1:
                sentiment.trainX = vstack([sentiment.trainX,unlabeled.X[index]])
                sentiment.trainy = np.concatenate((sentiment.trainy, y))
        print("len: ", len(used))
        if len(used) -prev == 0:
            break
        prev = len(used)
        cls = semiClassify.train_classifier(sentiment.trainX, sentiment.trainy,False)
        print(cls.densify())

    print("\nEvaluating semi-supervised classifier")
    semiClassify.evaluate(sentiment.trainX, sentiment.trainy, cls, 'train')
    semiClassify.evaluate(sentiment.devX, sentiment.devy, cls, 'dev')


if __name__ == "__main__":
    print("Reading data")
    tarfname = "data/sentiment.tar.gz"
    sentiment = read_files(tarfname)
    print("\nTraining classifier")
    import semiClassify
    cls = semiClassify.train_classifier(sentiment.trainX, sentiment.trainy,False)
    print("\nEvaluating")
    semiClassify.evaluate(sentiment.trainX, sentiment.trainy, cls, 'train')
    semiClassify.evaluate(sentiment.devX, sentiment.devy, cls, 'dev')

    features = sentiment.count_vect.get_feature_names()
    print("Writing predictions to a file")
    unlabeled = read_unlabeled(tarfname, sentiment)
    write_pred_kaggle_file(unlabeled, cls, "data/sentiment-pred.csv", sentiment)
    
    #write_basic_kaggle_file("data/sentiment-unlabeled.tsv", "data/sentiment-basic.csv")

    # You can't run this since you do not have the true labels
    # print "Writing gold file"
    # write_gold_kaggle_file("data/sentiment-unlabeled.tsv", "data/sentiment-gold.csv")
