'''
Common text preprocessing methods
'''

import re
import sys
from . import util
from .replacer import replacer

_to_remove = [
    '.', ',', '!', '?', ':', ';', '>', '<',
    '"', "'", '(', ')', '{', '}', '[', ']',
    '\\', '--', '`',
]
_to_substitute = util.flatten([_to_remove, [
    '-'
]])
_removal_pattern = replacer.prepare(_to_remove, onlyAtEnds=True)
_substitution_pattern = replacer.prepare(_to_substitute, onlyAtEnds=False)

_generic_digit_normalizer = (
    r'^[0-9]{1,}(\.[0-9]{1,}){0,1}$', '[DIGITS]'
)
_money_normalizer = (
    r'^\$[0-9]{1,}(,[0-9]{3})*(\.[0-9]{1,}){0,1}$', '[MONEY]'
)
_phone_normalizer = (
    r'(?<!\$)'
     '((\(?\d{3}\)|\d{3})[ \-\.]{0,1})?'
     '\d{3}[ \-\.]{0,1}\d{4}', '[PHONE]'
)
_phone_ext_normalizer = (
    r'^ext.?\d{1}-?\d{2,4}$|'
     '^x\d{1}-?\d{2,4}$|'
     '^\d{1}-\d{4}$',                     '[EXT]'
)

_url_normalizer = (
    r'^(https?|s?ftp)://[^\s]+$', '[URL]'
)
_email_normalizer = (
    r'^(mailto:)?([a-zA-Z0-9\.\-_]+\@[a-zA-Z0-9\.\-]+)(\.net|\.com|\.org|\.gov|\.edu|\.co\.uk)', '[EMAIL]'
)

def tokenize(line, clean=True, tolower=True, splitwords=False):
    tokens = line.strip().split()
    if clean:
        cleanTokens = []
        for token in tokens:
            token = token.strip()
            # only force UTF-8 encoding if still in Python 2
            if sys.version[0] == '2':
                token = token.encode('utf-8')
            token = replacer.remove(_removal_pattern, token)
            if tolower: token = token.lower()
            if splitwords:
                token = replacer.suball(_substitution_pattern, ' ', token)
                cleanTokens.extend(token.split())
            else:
                cleanTokens.append(token)
        tokens = cleanTokens
    return tokens

def normalizeNumeric(text, generic=True, money=True, phone=True):
    '''Takes as input a string or a sequence of tokens and returns the
    same string or token sequence.

    Three types of replacement are supported, all enabled by default:
        money   :: US monetary figures (e.g., $123.45) replaced by [MONEY]
        phone   :: US phone numbers (e.g., 800-867-5309) replaced by [PHONE]
        generic :: all other number sequences replaced by [DIGITS]

    The regexes aren't perfect, so take the output with a grain of salt.
    '''
    if phone:
        text = normalizePhone(text, extensions=True)

    def _normalizer(tokens, generic, money):
        normalized = []
        for t in tokens:
            # get the list of normalizations, based on specified arguments
            normalizers = []
            if generic: normalizers.append(_generic_digit_normalizer)
            if money: normalizers.append(_money_normalizer)

            for (ptrn, sub) in normalizers:
                t = re.sub(ptrn, sub, t)
            normalized.append(t)
        return normalized
    return _normalizeWrapper(text, _normalizer, generic, money)

def normalizeURLs(text):
    '''Takes as input a string or a sequence of tokens and returns the
    same string or token sequence, with all HTTP(S)/(S)FTP URLs replaced
    by [URL].

    This only recognizes URLs that start with a protocol (http://, ftp://, etc.);
    for example, www.google.com will not be normalized.
    '''
    def _normalizer(tokens):
        normalized = []
        for t in tokens:
            normalized.append(re.sub(_url_normalizer[0], _url_normalizer[1], t))
        return normalized
    return _normalizeWrapper(text, _normalizer)

def normalizeEmail(text):
    '''Takes as input a string or a sequence of tokens and returns the
    same string or token sequence, with detected email addresses replaced
    by [EMAIL].

    This only recognizes emails with alphanumeric, '.', and '-', and .com,
    .net, or .org TLDs.
    '''
    def _normalizer(tokens):
        normalized = []
        for t in tokens:
            normalized.append(re.sub(_email_normalizer[0], _email_normalizer[1], t))
        return normalized
    return _normalizeWrapper(text, _normalizer)

def normalizePhone(text, extensions=True):
    '''Takes as input a string or a sequence of tokens and returns the
    same string or token sequence, with all phone numbers replaced
    by [PHONE].

    If extensions=True, also replaces all phone extensions by [EXT].

    As always, the regexes aren't perfect.  Handle with skepticism.
    '''
    already_did_phone = False
    if type(text) == str:
        # if normalizing a string, normalize phone numbers here, as they
        # may include spaces
        text = re.sub(_phone_normalizer[0], _phone_normalizer[1], text)
        already_did_phone = True

    def _normalizer(tokens, extensions, already_did_phone):
        normalized = []
        for t in tokens:
            if not already_did_phone: t = re.sub(_phone_normalizer[0], _phone_normalizer[1], t)
            if extensions: t = re.sub(_phone_ext_normalizer[0], _phone_ext_normalizer[1], t)
            normalized.append(t)
        return normalized
    if (not already_did_phone) or extensions:
        text = _normalizeWrapper(text, _normalizer, extensions, already_did_phone)

    return text

def _normalizeWrapper(text, method, *args, **kwargs):
    if type(text) is str:
        tokens = text.split()
    elif type(text) in [list,tuple]:
        tokens = text

    normalized = method(tokens, *args, **kwargs)

    if type(text) == str:
        return ' '.join(normalized)
    else:
        return normalized

if __name__=='__main__':
    def _tests():
        header = lambda m: print('\n--- %s tests ---' % m)
        def test(inp, func, *args, **kwargs):
            out = func(inp, *args, **kwargs)
            print('Input: %s    Output: %s' % (inp, out))

        header('Phone')
        test('(867) 867-5309', normalizePhone)
        test('123-456-7890', normalizePhone)
        test('1234567 1234567890 12345 (77) 679 6792 111.111.1111', normalizePhone)
        test('123 $2378', normalizePhone)

        header('Money')
        test('$3,476 3,476 $99,99 $99,999.99 $123587.87 $12$7689.34', normalizeNumeric, money=True, generic=False, phone=False)

        header('Email')
        test('joe.blow@enron.com', normalizeEmail)
        test('joe_blow_the-third@enron.gov', normalizeEmail)
        test('this_should_fail@fail.place so@should the?%$se@google.com', normalizeEmail)
        test('this-sh0uld_b3.FINE@google-hat.org', normalizeEmail)
        test('mailto:this-sh0uld_b3.FINE@google-hat.subdomain.edu', normalizeEmail)
    _tests()
