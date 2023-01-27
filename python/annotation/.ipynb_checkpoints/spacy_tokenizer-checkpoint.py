import re
import spacy

from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc

try:
    nlp_model = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
except Exception as e:
    import sys
    import subprocess

    print("Language model not found. Downloading...")
    subprocess.run(
        ["python3", "-m", "spacy", "download", "--quiet", "en_core_web_sm"],
        # ["python3", "-m", "spacy", "download", "en_core_web_sm"],
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=True,
    )
    print("You have downloaded the language model. Please run it again.")
    sys.exit(-1)


def custom_tokenizer(nlp_model):
    """Tokenizer    https://spacy.io/usage/linguistic-features
    Args:
        nlp_model (Language): spacy language model

    Returns:
        Tokenizer: tokenizer
    """
    special_cases = {":)": [{"ORTH": ":)"}]}
    # 20210318 Δ:\u0394 20220325 \u0040, \u00AD
    prefix_re = re.compile(
        r'''^[\u0021-\u002D\u002F\u003A-\u0040\u005B-\u0060\u007B-\u00AF\u00B1\u00D7\u2010-\u21FF
        \u2200-\u2AFF\u3008\u3009\u0394]''',
        re.VERBOSE
    )
    # insert period:\u002E, °:\u00B0 for °C
    suffix_re = re.compile(
        r'''[\u0021-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u00AF\u00B0\u00B1\u00D7\u2010-\u21FF
        \u2200-\u2AFF\u3008\u3009\xad]$''',
        re.VERBOSE
    )
    #
    infix_re = re.compile(
        r'''[\u0021-\u002D\u002F\u003A-\u0040\u005B-\u0060\u007B-\u00AF\u00B1\u00D7\u2010-\u21FF
        \u2200-\u2AFF\u3008\u3009\xad]''',
        re.VERBOSE
    )
    simple_url_re = re.compile(r'''^https?://''')

    return Tokenizer(
        nlp_model.vocab, rules=special_cases,
                         prefix_search=prefix_re.search,
                         suffix_search=suffix_re.search,
                         infix_finditer=infix_re.finditer,
                         token_match=simple_url_re.match,
    )


def tokenize(text):
    sp_text = re.sub(r'[\s]', '\u0020', text)
    # Add 20210322
    sp_text = sp_text.replace('\xad', '-')
    nlp_model.tokenizer = custom_tokenizer(nlp_model)
    return nlp_model(sp_text)


class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        """https://spacy.io/usage/linguistic-features#custom-tokenizer-example
        initialier

        Args:
            vocab (Vocab): spacy vocab
        """
        self.vocab = vocab

    def __call__(self, text):
        """called as a function

        Args:
            text (str): text

        Returns:
            Doc: document
        """
        words = text.split(' ')
        # added for sequential spaces, check required
        words = [w if w else ' ' for w in words]
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


def tokenize_by_ws(text):
    nlp_model.tokenizer = WhitespaceTokenizer(nlp_model.vocab)
    return nlp_model(text)
