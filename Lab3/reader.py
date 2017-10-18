__author__ = "Marcus Rodan"

def read_sentences(file):
    return open(file).read().strip().split('\n\n')

def split_rows(sentences, column_names):
    return list(map(lambda sentence: list(map(lambda e: dict(zip(column_names, e.split(' '))),sentence.split('\n'))), sentences))

def read_formatted(file, column_names):
    return split_rows(read_sentences(file), column_names)
