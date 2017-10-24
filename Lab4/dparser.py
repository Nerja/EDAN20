import conll

__author__ = "Marcus Rodan"

def extract_subj_verb_cnt(formatted):
    subj_verb_cnt = {}
    for sentence in formatted:
        for w in sentence:
            if w['deprel'] == 'SS':
                verb = sentence[int(w['head'])]['form']
                subj = w['form']
                pair = (subj, verb)
                subj_verb_cnt[pair] = subj_verb_cnt.get(pair, 0) + 1
    return subj_verb_cnt

def sen_subj_verb_obj_cnt(sentence, subj_verb_obj_cnt):
    subjects    = [s for s in sentence if s['deprel'] == 'SS']
    objects     = [o for o in sentence if o['deprel'] == 'OO']
    for s in subjects:
        for o in objects:
            if s['head'] == o['head']:
                subj = s['form']
                obj  = o['form']
                head = sentence[int(s['head'])]
                verb = head['form']
                pair = (subj, verb, obj)
                subj_verb_obj_cnt[pair] = subj_verb_obj_cnt.get(pair, 0) + 1

def extract_subj_verb_obj_cnt(formatted):
    subj_verb_obj_cnt = {}
    for sentence in formatted:
        sen_subj_verb_obj_cnt(sentence, subj_verb_obj_cnt)
    return subj_verb_obj_cnt

def print_bindings(count_dict):
    print("Found {}".format(sum(count_dict.values())))
    sorted_keys     = sorted(count_dict, key=count_dict.get, reverse=True)
    sorted_pairs    = list(map(lambda k: (k, count_dict[k]), sorted_keys))
    top_pairs_str   = ''.join([str(k) + ' -> ' + str(v) + '\n' for k, v in sorted_pairs[:5]])
    print(top_pairs_str)

if __name__ == "__main__":
    train_file      = 'swedish_talbanken05_train.conll'
    column_names    = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'head', 'deprel', 'phead', 'pdeprel']
    sentences       = conll.read_sentences(train_file)
    formatted       = conll.split_rows(sentences, column_names)
    for sentence in formatted:
        for w in sentence:
            w['form'] = w['form'].lower()

    subj_verb_pairs_cnt = extract_subj_verb_cnt(formatted)
    print('Subject-verb pairs:')
    print_bindings(subj_verb_pairs_cnt)

    subj_verb_obj_cnt = extract_subj_verb_obj_cnt(formatted)
    print('Subject-Verb-Object pairs:')
    print_bindings(subj_verb_obj_cnt)
