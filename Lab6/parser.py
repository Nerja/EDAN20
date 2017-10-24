import sys
sys.path.insert(0, '../Lab5/')

import transition
import conll
import features
import pickle
from sklearn.externals import joblib

__author__ = "Marcus Rodan"

def read_blind_test():
    test_file = 'swedish_talbanken05_test_blind.conll'

    test_sentences = conll.read_sentences(test_file)
    column_names = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats']
    return conll.split_rows(test_sentences, column_names)

def load_classifier():
    classifier  = joblib.load("../Lab5/classifier_deprel_True_set3.pkl")
    vec         = pickle.load(open('../Lab5/vec_deprel_True_set3.pkl', 'rb'))
    return classifier, vec

def parse_ml(stack, queue, graph, trans):
    if stack and trans[:2] == 'ra':
        stack, queue, graph = transition.right_arc(stack, queue, graph, trans[3:])
        picked_trans = trans
    elif stack and trans[:2] == 'la':
        stack, queue, graph = transition.left_arc(stack, queue, graph, trans[3:])
        picked_trans = trans
    elif stack and trans[:2] == 're':
        stack, queue, graph = transition.reduce(stack, queue, graph)
        picked_trans = 're'
    else:
        stack, queue, graph = transition.shift(stack, queue, graph)
        picked_trans = 'sh'
    return stack, queue, graph, picked_trans

def predict_head_deprel_sent(sentence, classifier, vec, feature_names):
    queue = list(sentence)
    stack = list()
    graph = {'heads':{}, 'deprels':{}}
    graph['heads']['0'] = '0'
    graph['deprels']['0'] = 'ROOT'
    while queue:
        feats               = features.extract(stack, queue, graph, feature_names, sentence)
        predicted_act       = classifier.predict(vec.transform(feats))[0]
        stack, queue, graph, deprel = parse_ml(stack, queue, graph, predicted_act)

    for i in range(len(sentence)):
        sentence[i]['head'] = graph['heads'].get(str(i), str(0))
        sentence[i]['deprel'] = graph['deprels'].get(str(i), str(0))
        sentence[i]['phead'] = '_'
        sentence[i]['pdeprel'] = '_'

def predict_head_deprel(test_formatted, classifier, vec, feature_names):
    for sentence in test_formatted:
        predict_head_deprel_sent(sentence, classifier, vec, feature_names)

def print_out_file(formatted):
    output_string = ''
    column_out = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'head', 'deprel', 'phead', 'pdeprel']
    for sentence in formatted:
        for w in sentence[1:]:
            output_values = [w[k] for k in column_out]
            output_string += '\t'.join(output_values) + '\n'
        output_string += '\n'
    open('out', 'w').write(output_string)

if __name__ == "__main__":
    #Load blind test set
    test_formatted = read_blind_test()

    feature_names = ['can_la', 'can_re', 'stack_0_w', 'stack_0_pos', 'queue_0_w', 'queue_0_pos', 'stack_1_w', 'stack_1_pos', 'queue_1_w', 'queue_1_pos', 'lex_stack_0_fw', 'pos_stack_0_fw']
    classifier, vec = load_classifier()

    predict_head_deprel(test_formatted, classifier, vec, feature_names)

    print_out_file(test_formatted)
