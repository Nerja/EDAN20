import reader
import baseline #for dump to file
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model

__author__ = "Marcus Rodan"

def extract_sentence_features(sentence, feature_names):
    padding     = {'form': 'BOS', 'pos': 'BOS', 'chunk': 'BOS'}
    sentence    = [padding,padding] + sentence + [padding,padding]
    features    = []
    for i in range(2,len(sentence)-2):
        word_window     = [sentence[j]['form'] for j in range(i-2, i+3)]
        pos_window      = [sentence[j]['pos'] for j in range(i-2, i+3)]
        chunk_window    = [sentence[j]['chunk'] for j in range(i-2, i)]
        features += [dict(zip(feature_names, word_window + pos_window + chunk_window))]
    return features

def encode_classes(train_classes):
    classes         = sorted(list(set(train_classes)))
    num_to_class    = dict(enumerate(classes))
    class_to_num    = dict(list(map(lambda p: (p[1], p[0]), num_to_class.items())))
    y               = list(map(lambda c: class_to_num[c], train_classes))
    return y, num_to_class, class_to_num

def train_classifier(train_data, feature_names):
    print('Extracting features ...')
    train_features  = sum(list(map(lambda sen: extract_sentence_features(sen, feature_names), train_data)),[])
    train_classes   = sum(list(map(lambda sen: list(map(lambda w: w['chunk'], sen)),train_data)),[])

    #Perform one hot encoding
    vec = DictVectorizer(sparse=True)
    X_train = vec.fit_transform(train_features)
    y_train, num_to_class, class_to_num = encode_classes(train_classes)

    print('Training classifier ...')
    classifier = linear_model.LogisticRegression(penalty='l2', dual=True, solver='liblinear')
    classifier.fit(X_train, y_train)

    return vec, classifier, num_to_class, class_to_num

def add_predict_chunks(test_sentences, classifier, feature_names, num_to_class, class_to_num):
    for sentence in test_sentences:
        ca2 = 'BOS'
        ca1 = 'BOS'
        sentence_features = extract_sentence_features(sentence, feature_names)
        for i in range(0, len(sentence)):
            entry_features = sentence_features[i]
            entry_features['ca2'] = ca2
            entry_features['ca1'] = ca1
            sentence[i]['pchunk'] = num_to_class[classifier.predict(vec.transform(entry_features))[0]]
            ca2 = ca1
            ca1 = sentence[i]['pchunk']

if __name__ == "__main__":
    #Load training data
    column_names = ['form', 'pos', 'chunk']
    train_file = 'train.txt'
    train_sentences = reader.read_formatted(train_file, column_names)

    #Train classifier
    feature_names = ['wb2', 'wb1', 'w', 'wa1', 'wa2', 'pb2', 'pb1', 'p', 'pa1', 'pa2', 'ca2', 'ca1']
    vec, classifier, num_to_class, class_to_num = train_classifier(train_sentences, feature_names)

    #Load test data
    test_file = 'test.txt'
    test_sentences = reader.read_formatted(test_file, column_names)

    #Predict
    add_predict_chunks(test_sentences, classifier, feature_names, num_to_class, class_to_num)
    baseline.dump_to_file(test_sentences, 'out_ml_chunker')
