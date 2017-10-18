import unigram_probability as up

__author__ = "Marcus Rodan"

def count_bigrams(tokens):
    bigrams = list(zip(tokens, tokens[1:]))
    bigram_count = {}
    for bigram in bigrams:
        bigram_count[bigram] = bigram_count.get(bigram, 0) + 1
    return bigram_count

def compute_cond_prob(bi, bigram_count, wa_cnt, wb_cnt, nbr_unigrams):
    if bi in bigram_count:
        return bigram_count[bi] / wa_cnt
    else:
        return wb_cnt / nbr_words

def compute_bigram_pos(sentence_tokens, unigram_count, nbr_words, bigram_count):
    prob = 1.0
    nbr_words = len(unigram_count)
    sentence_bigrams = list(zip(sentence_tokens, sentence_tokens[1:]))
    print(''.join(map(lambda x:'=', range(70))))
    print("wi\twi+1\tCi,i+1\tC(i)\tP(wi+1|wi)")
    print(''.join(map(lambda x:'=', range(70))))
    for bi in sentence_bigrams:
        wa      = bi[0]
        wb      = bi[1]
        wa_cnt  = unigram_count[wa]
        wb_cnt  = unigram_count[wb]
        pwbwa   = compute_cond_prob(bi, bigram_count, wa_cnt, wb_cnt, nbr_words)
        prob *= pwbwa
        print("{}\t{}\t{}\t{}\t{}".format(wa, wb, bigram_count.get(bi, 0), wa_cnt, pwbwa))
    up.print_stats(prob, sentence_tokens)

if __name__ == "__main__":
    corpus_tokens, sentence = up.load_parameters()
    unigram_count   = up.count_unigrams(corpus_tokens)
    nbr_words       = len(corpus_tokens)
    bigram_count    = count_bigrams(corpus_tokens)
    sentence_tokens = ["<s>"] + up.tokenize(sentence) + ["</s>"]
    compute_bigram_pos(sentence_tokens, unigram_count, nbr_words, bigram_count)
