import re

START = '__START__'
STOP = '__STOP__'
UNK = '__UNK__'

TAGS = {
    'N': 'Common noun',
    'O': 'Pronoun',
    'S': 'nominal possessive',
    '^': 'proper noun',
    'Z': 'proper noun possessive',
    'L': 'nominal + verbal',
    'M': 'proper nound + verbal', # Mark'll
    'V': 'verb',
    'A': 'adjective',
    'R': 'adverb',
    '!': 'interjection', # lol, haha
    'D': 'determiner', # the, its
    'P': 'pre/post position', # while, to, for, with
    '&': 'coordinating conjunction', # and, but
    'T': 'verb particle', # out, off, but
    'X': 'existential there, predeterminers', # both, neither
    'Y': 'X + verbal', #there's, all's
    '#': 'hashtag',
    '@': 'at mention',
    '~': 'discourse marker',
    'U': 'URL/email',
    'E': 'emoticon',
    '$': 'numeral',
    ",": 'punctuation',
    'G': 'other abbreviations', # wby (what about you), idgaf
    START: 'start',
    STOP: 'stop'
}

# MANUAL_TAGS = ['#', '@', 'U', '$', ',']

# def get_manual_tag(word):
#     if word.startswith('#'):
#         return '#' # hashtag
#     elif word.startswith('@'):
#         return '@'  # mention
#     elif word.startswith('http') or word.startswith('www') or '.co' in word:
#         return 'U'  # url
#
#     # maybe remove more chars found in words?
#     clean_word = clean_word = re.sub('[^0-9a-zA-Z ]+', '', word)
#     if not clean_word.isalnum():
#         return ','  # punctuation
#     if not clean_word.isalpha():
#         return '$'  # numeric
#
#     # Don't have a solid rule for this one
#     return None