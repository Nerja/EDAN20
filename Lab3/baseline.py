import reader

__author__ = "Marcus Rodan"

def count_pos_chunk(train_data):
    pos_chunk = {}
    for sentence in train_data:
        for w in sentence:
            pos         = w['pos']
            chunk       = w['chunk']
            pos_dict    = pos_chunk.get(pos, {})
            pos_dict[chunk] = pos_dict.get(chunk, 0) + 1
            pos_chunk[pos] = pos_dict
    return pos_chunk

#Classifier classifying POS TAG -> CHUNK TAG
def train_classifier(train_data):
    pos_chunk_cnt       = count_pos_chunk(train_data)
    pos_chunk_mapping   = {}
    for pos, pos_dict in pos_chunk_cnt.items():
        pos_chunk_mapping[pos] = max(pos_dict, key=pos_dict.get)

    return pos_chunk_mapping

def add_predict_chunks(sentences, classifier):
    for sentence in sentences:
        for w in sentence:
            w['pchunk'] = classifier[w['pos']]

def dump_to_file(test_sentences, file):
    output_string = ''
    for sentence in test_sentences:
        for w in sentence:
            output_string += "{} {} {} {}\n".format(w['form'], w['pos'], w['chunk'], w['pchunk'])
        output_string += '\n'
    open(file, 'w').write(output_string)

if __name__ == "__main__":
    #Load training data
    column_names = ['form', 'pos', 'chunk']
    train_file = 'train.txt'
    train_sentences = reader.read_formatted(train_file, column_names)

    #Train classifier
    classifier = train_classifier(train_sentences)

    #Load test data
    test_file = 'test.txt'
    test_sentences = reader.read_formatted(test_file, column_names)

    #Predict
    add_predict_chunks(test_sentences, classifier)
    dump_to_file(test_sentences, 'out_base')
