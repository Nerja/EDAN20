import regex
import sys

__author__ = "Marcus Rodan"

def transform_sentence(sentence_match):
    sentence = sentence_match.group(1)
    return ' <s> ' + regex.sub(r'\p{P}', "", sentence.lower()) + ' </s> '

if __name__ == "__main__":
    text    = sys.stdin.read()
    text    = regex.sub(r'(\p{Lu}[^\.]*)\.', transform_sentence, text)
    print(text)
