import transition
import conll
import sys
import features
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
from sklearn.externals import joblib
from sklearn import metrics

"""
Gold standard parser
"""
__author__ = "Pierre Nugues | Marcus Rodan"

def reference(stack, queue, graph, use_deprel):
    """
    Gold standard parsing
    Produces a sequence of transitions from a manually-annotated corpus:
    sh, re, ra.deprel, la.deprel
    :param stack: The stack
    :param queue: The input list
    :param graph: The set of relations already parsed
    :return: the transition and the grammatical function (deprel) in the
    form of transition.deprel
    """
    # Right arc
    if stack and stack[0]['id'] == queue[0]['head']:
        # print('ra', queue[0]['deprel'], stack[0]['cpostag'], queue[0]['cpostag'])
        deprel = '.' + queue[0]['deprel']
        stack, queue, graph = transition.right_arc(stack, queue, graph)
        if use_deprel:
            return stack, queue, graph, 'ra' + deprel
        else:
            return stack, queue, graph, 'ra'
    # Left arc
    if stack and queue[0]['id'] == stack[0]['head']:
        # print('la', stack[0]['deprel'], stack[0]['cpostag'], queue[0]['cpostag'])
        deprel = '.' + stack[0]['deprel']
        stack, queue, graph = transition.left_arc(stack, queue, graph)
        if use_deprel:
            return stack, queue, graph, 'la' + deprel
        else:
            return stack, queue, graph, 'la'
    # Reduce
    if stack and transition.can_reduce(stack, graph):
        for word in stack:
            if (word['id'] == queue[0]['head'] or
                        word['head'] == queue[0]['id']):
                # print('re', stack[0]['cpostag'], queue[0]['cpostag'])
                stack, queue, graph = transition.reduce(stack, queue, graph)
                return stack, queue, graph, 're'
    # Shift
    # print('sh', [], queue[0]['cpostag'])
    stack, queue, graph = transition.shift(stack, queue, graph)
    return stack, queue, graph, 'sh'

def extract_features_classes_sent(sentence, use_deprel, feature_names):
    sent_features   = []
    sent_classes    = []
    queue           = list(sentence)
    stack           = []
    graph = {}
    graph['heads'] = {}
    graph['heads']['0'] = '0'
    graph['deprels'] = {}
    graph['deprels']['0'] = 'ROOT'
    while queue:
        curr_features                   = features.extract(stack, queue, graph, feature_names, sentence)
        stack, queue, graph, curr_class = reference(stack, queue, graph, use_deprel)
        sent_features   += [curr_features]
        sent_classes    += [curr_class]
    return sent_features, sent_classes

def extract_features_classes(formatted_corpus, use_deprel, feature_names):
    features = []
    classes  = []
    for sentence in formatted_corpus:
        sent_features, sent_classes = extract_features_classes_sent(sentence, use_deprel, feature_names)
        features += sent_features
        classes += sent_classes
    return features, classes

def load_params():
    if(len(sys.argv) != 4):
        print('Params should be usedeperel[True/False] fset[1,2,3] train[True/False]')
        sys.exit()
    use_deprel  = sys.argv[1] == 'True'
    f_set       = int(sys.argv[2])
    train       = sys.argv[3] == 'True'
    print("Using settings: use_deprel: {}, feature_set: {}, train: {}".format(use_deprel, f_set, train))
    return use_deprel, f_set, train

if __name__ == '__main__':
    use_deprel, f_set, train = load_params()

    train_file = 'swedish_talbanken05_train.conll'
    test_file = 'swedish_talbanken05_test.conll'
    column_names_2006 = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'head', 'deprel', 'phead', 'pdeprel']

    sentences = conll.read_sentences(train_file)
    formatted_corpus = conll.split_rows(sentences, column_names_2006)

    # Select features to use
    if f_set == 1:
        feature_names = ['can_la', 'can_re', 'stack_0_w', 'stack_0_pos', 'queue_0_w', 'queue_0_pos']
    elif f_set == 2:
        feature_names = ['can_la', 'can_re', 'stack_0_w', 'stack_0_pos', 'queue_0_w', 'queue_0_pos', 'stack_1_w', 'stack_1_pos', 'queue_1_w', 'queue_1_pos']
    else:
        feature_names = ['can_la', 'can_re', 'stack_0_w', 'stack_0_pos', 'queue_0_w', 'queue_0_pos', 'stack_1_w', 'stack_1_pos', 'queue_1_w', 'queue_1_pos', 'lex_stack_0_fw', 'pos_stack_0_fw']

    use_deprel = True
    train_features, train_classes = extract_features_classes(formatted_corpus, use_deprel, feature_names)

    # One hot encode features
    vec     = DictVectorizer(sparse=True)
    X_train = vec.fit_transform(train_features)
    y_train = train_classes #fit will transform from string to numeric

    # Train classifier(and save it) or load it
    classifier = linear_model.LogisticRegression(penalty='l2', dual=True, solver='liblinear')
    file_name = "_deprel_{}_set{}.pkl".format(use_deprel, f_set)
    print("Filename: {}".format(file_name))
    if train:
        print('Training model ...')
        model = classifier.fit(X_train, y_train)
        joblib.dump(classifier, "classifier" + file_name)
        joblib.dump(vec, "vec" + file_name)
    else:
        print('Loading model ...')
        classifier = joblib.load("classifier" + file_name)

    #Load testset and predict
    test_sentences              = conll.read_sentences(test_file)
    test_formatted              = conll.split_rows(sentences, column_names_2006)
    test_features, test_classes = extract_features_classes(test_formatted, use_deprel, feature_names)
    X_test                      = vec.transform(test_features)
    predicted_classes           = classifier.predict(X_test)

    print("Classification report for classifier %s:\n%s\n"
           % (classifier, metrics.classification_report(test_classes, predicted_classes)))

    nbr_correct = sum([1 if predicted_classes[i] == test_classes[i] else 0 for i in range(len(test_classes))])
    print("Test accuracy: {}".format(nbr_correct/len(test_classes)))
