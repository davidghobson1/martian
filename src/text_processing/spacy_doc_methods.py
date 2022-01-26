"""spacy_doc_methods

This module provides functions to help automate text processing with spaCy 
Functions include noun phrase retrieval, processing with lemmatization and stopword removal,
entity retrieval, and basic tasks like tokenization.

Filename: spacy_doc_methods.py
Maintainers: David Hobson, Saruggan Thiruchelvan
Last Updated: Decemeber 8, 2021
"""

import numpy as np

import spacy
import nltk

default_noun_phrase_format = r"""NP: {<JJ.*>*<NN.*>+<DT>*<IN.*>*<JJ.*>*<NN.*>+}
                      {<JJ.*>+<NN.*>+}"""
   
def tokenize(doc, to_lowercase=False, lemmatize=False):
    """
    Get the tokens from a given sentence.
    tokenize(doc, to_lowercase, lemmatize)

    doc: spaCy Doc object
        The doc object for the sentence to be tokenized.
    to_lowercase: bool; optional (default = False)
        True if tokens are to be converted to lowercase.
    lemmatize: bool; optional (default = False)
        True if tokens are to be lemmatized. Note that if lemmatize is True, tokens are converted to lowercase by default
        with the exception of proper nouns (i.e. names and places) (this is the convention in spaCy).
        Setting to_lowercase=True and lemmatize=True, will convert everything to lowercase (even proper nouns)
    
    Returns [str]; The list of tokens for the given sentence.
    Note: If the given sentence is not a string, the numpy nan is returned.
    This is useful for processing on a Pandas DataFrame without worrying about types.
    """
    
    if not isinstance(doc, spacy.tokens.Doc):
        return np.nan
    
    if lemmatize:
        tokens = [token.lemma_ for token in doc]
    elif to_lowercase:
        tokens = [token.lower_ for token in doc]
    else:
        tokens = [token.text for token in doc]
    
    return tokens

def get_all_pos(doc, to_lowercase=False, lemmatize=False, universal_tag=False):
    """
    Get all the possible pos_tuples (words paired with their corresponding part-of-speech)
    for the given sentence.
    get_all_pos(doc, to_lowercase, lemmatize, universal_tag)

    doc: spaCy Doc object
        The doc object for the sentence from which the pos-tuples will be derived.
    to_lowercase: bool; optional (default = True)
        True if pos_tuple tokens should be converted to lowercase.
    lemmatize: bool; optional (default = False)
        True if pos_tuple tokens should be lemmatized. 
        Note that if lemmatize is True, tokens are converted to lowercase by default with the exception of proper nouns (i.e. names and places). This is the convention in spaCy.
        Setting to_lowercase=True with lemmatize=True, will convert everything to lowercase (even proper nouns)
    universal_tag: bool; optional (default = False)
        True if pos-tags should follow universal format (see: https://universaldependencies.org/u/pos/) instead of the detailed tag format 
        In universal format, nouns are given by the tag 'NOUN' instead of the tag 'NN'
                             verbs by 'VERB' instead of the tag 'VB',
                             etc.
                             Please refer to the link for more details.

    Returns [(str, str)]; The list of pos_tuples derived from a given sentence.
    That is, the list of tuples consisting of each word paired with its part-of-speech
    """
    
    if not isinstance(doc, spacy.tokens.Doc):
        return np.nan
    
    # nested if statements are for computational efficiency, as opposed to code readability
    if lemmatize:                             
        if universal_tag:
            if to_lowercase:
                pos_tuples = [(token.lemma_.lower(), token.pos_) for token in doc]
            else:
                pos_tuples = [(token.lemma_, token.pos_) for token in doc]
        else:
            if to_lowercase:
                pos_tuples = [(token.lemma_.lower(), token.tag_) for token in doc]
            else:
                pos_tuples = [(token.lemma_, token.tag_) for token in doc]
    elif to_lowercase:
        if universal_tag:
            pos_tuples = [(token.lower_, token.pos_) for token in doc]
        else:
            pos_tuples = [(token.lower_, token.tag_) for token in doc]
    else:
        if universal_tag:
            pos_tuples = [(token.text, token.pos_) for token in doc]
        else:
            pos_tuples = [(token.text, token.tag_) for token in doc]
        
    return pos_tuples

def get_pos(doc, tag, to_lowercase=False, lemmatize=False):
    """
    Get all tokens corresponding to a specific part-of-speech tag for the given sentence.
    Note that the given tag must either match or partially match the detailed pos-tag. (Not the universal tag!)
    (For example, to search for adjectives you need to specify tag="JJ" or just tag="J", etc.
     searching tag='ADJ' won't work since it's a universal tag, not a detailed pos-tag)
    get_pos(doc, tag, to_lowercase, lemmatize)

    doc: spaCy Doc object
        The doc object for the sentence to be searched
    tag: str
        The tag associated with the part-of-speech. Note that the given tag must either match or partially match a detailed pos-tag
    to_lowercase: bool; optional (default = False)
        True if tokens should be converted to lowercase.
    lemmatize: bool; optional (default = False)
        True if tokens should be lemmatized.
        Note that if lemmatize is True, tokens are converted to lowercase by default with the exception of proper nouns (i.e. names 
        and places). This is the convention in spaCy.
        Setting to_lowercase=True with lemmatize=True, will convert even proper nouns to lowercase (i.e. everything to lowercase)

    Returns [str]; The list of tokens corresponding to the given part-of-speech tag.
    """

    if not isinstance(doc, spacy.tokens.Doc):
        return np.nan
    
    tokens = [token for token in doc if tag in token.tag_]
    
    if lemmatize:
        if to_lowercase:
            tokens = [token.lemma_.lower() for token in tokens]
        else:
            tokens = [token.lemma_ for token in tokens]
    elif to_lowercase:
        tokens = [token.lower_ for token in tokens]
    
    return tokens


def get_noun_phrases(
        doc,
        noun_phrase_format=default_noun_phrase_format, 
        to_lowercase=False, 
        singularize=False
):
    """
    Derive all the noun phrases contained in a given sentence.
    get_noun_phrases(doc, noun_phrase_format, to_lowercase, singularize)

    doc: spaCy Doc object
        The doc object for the sentence to derive noun phrases from.
    noun_phrase_format: str; optional (default = TextProcessor.noun_phrase_format)
        A string specifying how the noun phrases should be formatted/structured.
    to_lowercase: bool; optional (default=False)
        True if noun phrases should be converted to lowercase
    singularize: bool; optional (default = False)
        True if individual nouns within noun phrases should be singularized.
        Note that if True, tokens are converted to lowercase by default with the exception of proper nouns (i.e. names 
        and places). This is the convention in spaCy.
        Setting to_lowercase=True with singularize=True, will convert even proper nouns to lowercase (i.e. everything to lowercase)

    Returns [str]; The list of noun phrases produced from the given sentence.
    Note: If no pos_tuples can be derived from the sentence, it will return np.nan. 
    This is useful for processing on a Pandas DataFrame without worrying about types.
    """
    
    if not isinstance(doc, spacy.tokens.Doc):
        return np.nan
    
    # get all pos tuples
    pos_tuples = get_all_pos(doc, to_lowercase=to_lowercase, lemmatize=singularize)

    # find the noun phrases based on the noun phrase format
    pos_tuples_noun_phrases = __build_noun_phrases(pos_tuples, noun_phrase_format)

    if not isinstance(pos_tuples_noun_phrases, list):
        return np.nan

    return [token for (token, tag) in pos_tuples_noun_phrases if tag == 'NP']

def __build_noun_phrases(
            pos_tuples, 
            noun_phrase_format=default_noun_phrase_format
    ):
    
    """
    <PRIVATE> Helper function to build the noun phrases by combining adjacent tuples that form a noun phrase. Returns the list of pos_tuples 
    with the noun phrases combined and labelled with the tag 'NP'
    __build_noun_phrases(pos_tuples, noun_phrase_format)

    pos_tuples: [(str, str)]
        The list of pos_tuples to derive noun phrases from. Tags in the tuples should be in detailed-tag format and not
        universal format
    noun_phrase_format: str; optional (default = TextProcessor.default_noun_phrase_format)
        A string specifying how the noun phrases should be formatted/structured.

    Returns [(str, str)]; A list of pos_tuples containing noun phrases produced from the
    original list of pos_tuples. The noun phrases are assigned the tag 'NP'
    Note: If pos_tuples is not a list, it will return np.nan. This is useful for processing on a
    Pandas DataFrame without worrying about types.
    """
        
    if not isinstance(pos_tuples, list):
        return np.nan

    chunk_parser = nltk.RegexpParser(noun_phrase_format)      # define the noun phrase parser
    parsed_sentence = chunk_parser.parse(pos_tuples)          # parse the sentence

    pos_tuples_noun_phrases = []
    for chunk in parsed_sentence:
        if isinstance(chunk, nltk.tree.Tree):                 # found a noun phrase to add
            noun_phrase = ""                                  # build the noun phrase
            for i in range(len(chunk)):
                if i == len(chunk) - 1:
                    noun_phrase += chunk[i][0]
                else:
                    noun_phrase += chunk[i][0] + " "
            pos_tuples_noun_phrases.append((noun_phrase, 'NP'))
        else:
            pos_tuples_noun_phrases.append(chunk)
    return pos_tuples_noun_phrases

def get_all_ents(doc, to_lowercase=False):
    """
    Get all the entity-tuples (words paired with their entity labels) for the given sentence.
    get_all_ents(doc, to_lowercase)

    doc: spaCy Doc object
        The doc object for the sentence where the entities will be derived from.
    to_lowercase: bool; optional (default = True)
        True if entity_tuple tokens should be converted to lowercase.

    Returns [(str, str)]; The list of entity_tuples derived from a given sentence.
    That is, the list of tuples consisting of each word paired with its entity-label
    """
    
    if not isinstance(doc, spacy.tokens.Doc):
        return np.nan

    if to_lowercase:
        entity_tuples = [(ent.text.lower(), ent.label_) for ent in doc.ents]
    else:
        entity_tuples = [(ent.text, ent.label_) for ent in doc.ents]
    
    return entity_tuples

def get_ent(doc, ent_label, to_lowercase=False):
    """
    Get the list of words corresponding to the given entity-label for the given sentence.
    get_all_ents(doc, to_lowercase)

    doc: spaCy Doc object
        The doc object for the sentence where the entities will be extracted from.
    ent_label: str
        The entity label to match for.
    to_lowercase: bool; optional (default = True)
        True if tokens should be converted to lowercase.

    Returns [str]; The list of tokens matching the given entity-label.
    """
    
    if not isinstance(doc, spacy.tokens.Doc):
        return np.nan
    
    if to_lowercase:
        ents = [ent.text.lower() for ent in doc.ents if ent.label_ == ent_label]
    else:
        ents = [ent.text for ent in doc.ents if ent.label_ == ent_label]
    
    return ents

def process(doc, to_lowercase=True, preserve_noun_phrases=False, remove_numbers=True, remove_urls=True, remove_emails=True):
    """
    Tokenize, lemmatize, and remove stopwords from a given sentence. Returns a list of tokens.
    Optionally convert tokens to lowercase, preserve noun phrases, remove numbers, urls, or emails.
    This method is intended for text pre-processing for machine learning and other AI algorithms such as
    topic modelling, sentiment analysis, etc.
    process(doc, to_lowercase, preserve_noun_phrases, remove_numbers, remove_urls, remove_emails)

    doc: spaCy Doc object
        The doc object of the string to be processed
    to_lowercase: bool; optional (default = True)
        True if tokens should be converted to lowercase.
    preserve_noun_phrases: bool; optional (default = False)
        True if noun phrases should be preserved in the list of tokens.
    remove_numbers: bool; optional (default = True)
        True if numbers/digits should be excluded from the list of tokens.
    remove_urls: bool; optional (default = True)
        True if urls should be excluded from the list of tokens.
    remove_emails: bool; optional (default = True)
        True if email addresses should be excluded from the list of tokens.
    
    Returns [str]; The list of lemmatized and non-stopword tokens from the given sentence.
    Note: If no pos_tuples can be derived from the sentence or if the sentence cannot be casted to,
    a string, it will return np.nan. This is useful for processing on a Pandas DataFrame without
    worrying about types.
    """
        
    if not isinstance(doc, spacy.tokens.Doc):
        return np.nan
    
    filtered_tokens = []
    if preserve_noun_phrases:
        filtered_tokens = __process_with_noun_phrases(doc, to_lowercase=to_lowercase, remove_numbers=remove_numbers, remove_urls=remove_urls, remove_emails=remove_emails)
    else:
        for token in doc:
            if is_meaningful_token(token, remove_numbers=remove_numbers, remove_urls=remove_urls, remove_emails=remove_emails):
                filtered_tokens.append(token.lemma_ if not to_lowercase else token.lemma_.lower())

    return filtered_tokens if len(filtered_tokens) > 0 else np.nan

def __process_with_noun_phrases(doc, to_lowercase=True, remove_numbers=True, remove_urls=True, remove_emails=True):
    """
    <PRIVATE> Helper function to process a sentence when noun phrases are to be preserved
    __process_with_noun_phrases(doc, to_lowercase, remove_numbers, remove_urls, remove_emails)

    doc: spaCy Doc object
        The doc object of the string to be processed
    to_lowercase: bool; optional (default = True)
        True if tokens should be converted to lowercase.
    remove_numbers: bool; optional (default = True)
        True if numbers/digits should be excluded from the list of tokens.
    remove_urls: bool; optional (default = True)
        True if urls should be excluded from the list of tokens.
    remove_emails: bool; optional (default = True)
        True if email addresses should be excluded from the list of tokens.
    
    Returns [str]; The list of lemmatized and non-stopword tokens and noun phrases from the given sentence.
    """  
    
    filtered_tokens = []
    
    # parse the noun phrases
    pos_tuples = get_all_pos(doc, to_lowercase=to_lowercase, lemmatize=True)       # get all the parts-of-speech
    chunk_parser = nltk.RegexpParser(default_noun_phrase_format)                   # define the noun-phrase parser
    chunks = chunk_parser.parse(pos_tuples)                                        # parse together the noun phrases

    # find all important tokens and noun phrases
    curr_token_index, curr_chunk_index = 0, 0
    while curr_token_index < len(doc):
        token = doc[curr_token_index]
        chunk = chunks[curr_chunk_index]

        if isinstance(chunk, nltk.tree.Tree):                                                      # found a noun phrase
            noun_phrase = ""                                 
            for i in range(len(chunk)):
                if i == len(chunk) - 1:
                    noun_phrase += chunk[i][0]
                else:
                    noun_phrase += chunk[i][0] + " "
            curr_token_index += i
            filtered_tokens.append(noun_phrase if not to_lowercase else noun_phrase.lower())
        else:                                                                                      # found a regular token
            if is_meaningful_token(token, remove_numbers=remove_numbers, remove_urls=remove_urls, remove_emails=remove_emails):
                filtered_tokens.append(token.lemma_ if not to_lowercase else token.lemma_.lower())

        curr_token_index += 1
        curr_chunk_index += 1
        
    return filtered_tokens

def is_meaningful_token(token, remove_numbers=True, remove_urls=True, remove_emails=True):
    """
    Indicates whether the given token should be included in the list of processed words.
    That is, the word must not be a stopword, a punctuation mark, and optionally not be a number,
    URL, or email address.
    is_token_meaningful(token, remove_numbers, remove_urls, remove_emails)

    token: spaCy Span object
        The span object of the token/string to be processed
    remove_numbers: bool; optional (default = True)
        True if numbers/digits should be excluded from the list of tokens.
    remove_urls: bool; optional (default = True)
        True if urls should be excluded from the list of tokens.
    remove_emails: bool; optional (default = True)
        True if email addresses should be excluded from the list of tokens.
    
    Returns bool; True if the token should be included in the list of processed words
    """     
    
    if token.is_stop or token.is_punct:
        return False
    elif remove_numbers and token.like_num:
        return False
    elif remove_urls and token.like_url:
        return False
    elif remove_emails and token.like_email:
        return False
    else: 
        return True
