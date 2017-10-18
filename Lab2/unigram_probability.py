import regex
import sys
import math

__author__ = "Marcus Rodan"

def tokenize(text):
    return regex.findall(r'\p{L}+|<s>|</s>', text)

def count_unigrams(words):
    unigram_count = {}
    for w in words:
        unigram_count[w] = unigram_count.get(w, 0) + 1
    return unigram_count

def print_stats(prob, sentence):
    entropy = -(1/len(sentence))*math.log2(prob)
    perplexity = math.pow(2, entropy)
    geometric_mean = math.pow(prob, 1/len(sentence))
    print(''.join(map(lambda x:'=', range(70))))
    print("Probability unigrams:\t\t{}".format(prob))
    print("Geometric mean Probability:\t{}".format(geometric_mean))
    print("Entropy rate:\t\t\t{}".format(entropy))
    print("Perplexity\t\t\t{}".format(perplexity))

def compute_unigram_pos(sentence, unigram_count, nbr_words):
    prob = 1.0
    print(''.join(map(lambda x:'=', range(70))))
    print("wi\tC(wi)\t#words\tP(wi)")
    print(''.join(map(lambda x:'=', range(70))))
    for w in sentence:
        wi   = w
        c_wi = unigram_count.get(wi, 0)
        p_wi = c_wi / nbr_words
        prob *= p_wi
        print("{}\t{}\t{}\t{}".format(wi, c_wi, nbr_words, p_wi))
    print_stats(prob, sentence)

def load_parameters():
    if(len(sys.argv) != 2):
        print("Must give sentence as argument!")
        sys.exit()

    corpus          = sys.stdin.read()
    corpus_tokens   = tokenize(corpus)
    return corpus_tokens, sys.argv[1].lower()

if __name__ == "__main__":
    corpus_tokens, sentence = load_parameters()
    unigram_count   = count_unigrams(corpus_tokens)
    nbr_words       = len(corpus_tokens)
    sentence_tokens = tokenize(sentence) + ["</s>"]
    compute_unigram_pos(sentence_tokens, unigram_count, nbr_words)
