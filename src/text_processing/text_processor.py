"""TextProcessor

This class is to help automate text processing for the analysis 
of unstructured text data to be used for text mining and NLP tasks.

The main NLP library used within this class is NLTK.

Filename: text_processor.py
Maintainers: David Hobson, Saruggan Thiruchelvan
Last Updated: December 7, 2021

Areas of potential issues:
- Nouns aren't separated when they have a "/" (ex. "examples/instances" isn't split into "examples", "/", "instances")
- Words containing '&' are separated even when this isn't a good thing (ex. "A&W" is split into 'A', '&', 'W')
"""

import nltk
from nltk import TweetTokenizer

import string
import re

import numpy as np

class TextProcessor:
    """TextProcessor
    
    This class is to help automate text processing for the analysis 
    of unstructured text data to be used for text mining and NLP tasks.
    
    The main NLP library used within this class is NLTK.
    """
    
    # specifies how noun phrases are collected by the noun phrase parser
    default_noun_phrase_format = r"""NP: {<JJ.*>*<NN.*>+<DT>*<IN.*>*<JJ.*>*<NN.*>+}
                              {<JJ.*>+<NN.*>+}"""

    punctuation = {char for char in string.punctuation} ; punctuation.add('...')

    def __init__(
        self, 
        tokenizer = nltk.word_tokenize, 
        lemmatizer = nltk.WordNetLemmatizer().lemmatize, 
        stopwords = nltk.corpus.stopwords.words('english')
    ):
        """
        Initialize a new TextProcessor object.
        __init__(self, tokenizer, lemmatizer, stopwords)

        tokenizer: str -> [str]; optional (default = nltk.tokenize.word_tokenize)
            Tokenizer function. Function that takes in a sentence string and returns 
            the list of token strings.
        lemmatizer: str -> str; optional (default = nltk.stem.WordNetLemmatizer().lemmatize)
            Lemmatizer function. Function that takes in a token string and returns the 
            lemmatized token string.
        stopwords: [str]; optional (default = nltk.corpus.stopwords.words('english))
            List of stopwords. The list of words to be ignored during processing.
        """
        
        self.tokenizer = tokenizer
        self.lemmatizer = lemmatizer
        self.stopwords = set(stopwords)

##############################################################################
## Stopword Methods
##############################################################################

    def add_stopwords(self, new_stopwords):
        """
        Add a word or list of words to the list of stopwords.
        add_stopwords(self, new_stopwords)

        new_stopwords: str or [str]
            A single word or a list of words to be added to the list of stopwords.
            
        Raises ValueError if a list entry is not a string. In this case, none of the 
        list entries are added to the stopwords list
        
        Returns: None
        """
        
        if isinstance(new_stopwords, str):
            self.stopwords.add(new_stopwords)
        elif isinstance(new_stopwords, list):
            for stopword in new_stopwords:            # check all entries are strings
                if not isinstance(stopword, str):
                    raise ValueError(f"A list entry (entry={stopword}) was found that wasn't a string. Only strings can be added as stopwords")
            for stopword in new_stopwords:            # add to the list of stopwords 
                self.stopwords.add(stopword)
                
    def remove_stopwords(self, old_stopwords):
        """
        Remove a word or list of words from the list of stopwords.
        remove_stopwords(self, old_stopwords)

        old_stopwords: str or [str]
            A single word or list of words to be removed from the list of stopwords.
            
        Returns: None
        """
    
        if isinstance(old_stopwords, str):
            try:
                self.stopwords.remove(old_stopwords)
            except(KeyError):
                pass
            return
        elif isinstance(old_stopwords, list):
            for stopword in old_stopwords:
                try:
                    self.stopwords.remove(stopword)
                except(KeyError):
                    pass

    def get_stopwords(self):
        """
        Get the list of stopwords.
        get_stopwords(self)

        Returns [str]; The list of stopwords. That is, the list of words that are ignored 
        during processing.
        """
        
        return self.stopwords

##############################################################################
## Tokenization Methods
##############################################################################

    def tokenize(self, sentence, to_lowercase=False, lemmatize=False):
        """
        Get the tokens from a given sentence.
        tokenize(self, sentence, to_lowercase, lemmatize)

        sentence: str
            The sentence to be tokenized.
        to_lowercase: bool; optional (default = False)
            True if tokens are to be converted to lowercase.
        lemmatize: bool; optional (default = False)
            True if tokens are to be lemmatized
        Returns [str]; The list of tokens for the given sentence.
        Note: If the given sentence is not a string, the numpy nan is returned.
        This is useful for processing on a Pandas DataFrame without worrying about types.
        """
        
        if not isinstance(sentence, str):
            return np.nan     
        
        if lemmatize:
            lemmatized_pos_tuples = self.get_all_pos(sentence, to_lowercase=to_lowercase, lemmatize=True)
            return [token for (token, tag) in lemmatized_pos_tuples]

        tokens = self.tokenizer(sentence)
        
        if to_lowercase:
            tokens = self.tokens_to_lowercase(tokens)

        return tokens

    def tokens_to_lowercase(self, tokens):
        """
        Convert each token in list of tokens to lowercase.
        tokens_to_lowercase(self, tokens)

        tokens: [str]
            List of tokens to be converted to lowercase.

        Returns [str]; The given list of tokens converted to lowercase.
        Note: If tokens is not a list, it will return np.nan. This is useful for processing on a 
        Pandas DataFrame without worrying about types.
        """

        if not isinstance(tokens, list):
            return np.nan
        return [token.lower() for token in tokens]
        

##############################################################################
## Parts-of-Speech (POS) Methods
##############################################################################

    def get_all_pos(
            self, 
            sentence, 
            to_lowercase=False, 
            lemmatize=False
    ):
        """
        Get all the possible pos_tuples (words paired with their corresponding part-of-speech)
        for the given sentence.
        get_all_pos(self, sentence, to_lowercase, lemmatize)

        sentence: str
            The sentence to derive the pos_tuples from.
        to_lowercase: bool; optional (default = True)
            True if pos_tuple tokens should be converted to lowercase.
        lemmatize: bool; optional (default = False)
            True if pos_tuple tokens should be lemmatized. 

        Returns [(str, str)]; The list of pos_tuples derived from a given sentence.
        That is, the list of tuples consisting of each word paired with its part-of-speech
        """
        
        sentence = str(sentence)
        if sentence == 'nan':
            return np.nan
        
        tokens = self.tokenize(sentence)     # tokenize
        pos_tuples = nltk.pos_tag(tokens)    # get the pos
         
        if lemmatize:
            pos_tuples = self.lemmatize_pos_tuples(pos_tuples)
        
        if to_lowercase:
            tokens = self.tokens_to_lowercase([token for (token, tag) in pos_tuples])
            tags = [tag for (token, tag) in pos_tuples]
            pos_tuples = [(tokens[i], tags[i]) for i in range(len(pos_tuples))] 

        return pos_tuples

    def get_pos(
            self, 
            sentence, 
            tag, 
            to_lowercase=False, 
            lemmatize=False
    ):
        """
        Get all tokens corresponding to a specific part-of-speech for the given sentence.
        Note that the given tag must either match or partially match a pos-tag in NLTK's tagset
        (For example, to search for adjectives you need to specify tag="JJ" or just tag="J", etc.
         search on Google for more information about NTLK's tagset)
        get_pos(self, sentence, tag, to_lowercase, lemmatize)

        sentence: str
            The sentence to derive the tokens from.
        tag: str
            The tag associated with the part-of-speech. Note that the given tag must either match 
            or partially match a pos-tag in NLTK's tagset
        to_lowercase: bool; optional (default = False)
            True if tokens should be converted to lowercase.
        lemmatize: bool; optional (default = False)
            True if tokens should be lemmatized.

        Returns [str]; The list of tokens corresponding to the given part-of-speech tag.
        """

        sentence = str(sentence)
        if sentence == 'nan':
            return np.nan

        pos_tuples = self.get_all_pos(sentence, to_lowercase, lemmatize)     

        return TextProcessor.__filter_pos_tuples(pos_tuples, tag)

    def __filter_pos_tuples(pos_tuples, match_tag):
        """
        <PRIVATE CLASS METHOD> Returns the tokens whose pos tag matches the given match_tag from the 
        given list of pos_tuples.
        __filter_pos_tuples(self, pos_tuples, match_tag)

        pos_tuples: [(str, str)]
            List of pos_tuples to filter from.
        match_tag: str
            The part-of-speech tag to filter the pos_tuples on.

        Returns [tokens]; The list of tokens that have the same tag as the given match_tag.
        Note: If pos_tuples is not a list, it will return np.nan. This is useful for processing on a 
        Pandas DataFrame without worrying about types.
        """

        if not isinstance(pos_tuples, list):
            return np.nan
        
        return [token for (token, tag) in pos_tuples if match_tag in tag]

##############################################################################
## Lemmatization Methods
##############################################################################

    def lemmatize_pos_tuples(self, pos_tuples):
        """
        Lemmatize the token part of each tuple in the given list of pos_tuples.
        lemmatize_pos_tuples(self, pos_tuples)

        pos_tuples: [(str, str)]
            The list of pos_tuples to be lemmatized.
        
        Returns [(str, str)]; The list of pos_tuples with the tokens lemmatized.
        """
        
        pos_tuples_wordnet = TextProcessor._TextProcessor__format_pos_tuples_to_wordnet(pos_tuples)
        
        lemmatized_pos_tokens = [self.lemmatizer(token, pos=tag) for (token, tag) in pos_tuples_wordnet]      # lemmatize the tokens
        original_pos_tags = [tag for (token, tag) in pos_tuples]                                              # keep the original POS tag (not the wordnet tag)

        # match each token with their original pos-tag
        return [(lemmatized_pos_tokens[i], original_pos_tags[i]) for i in range(len(pos_tuples))]
    
    def __format_pos_tuples_to_wordnet(pos_tuples):
        """
        <PRIVATE CLASS METHOD> Convert the pos-tags from the given a list of pos_tuples, to the format 
        that is accepted by WordNet.
        __format_pos_tuples_to_wordnet(pos_tuples)

        pos_tuples: [(str, str)]
            List of pos_tuples to be formatted.

        Returns [(str, str)]; The pos-tuples with WordNet-compatable pos-tags
        """

        # dictionary of the WordNet POS labels
        wordnet_tags = {"J": nltk.corpus.wordnet.ADJ,
                "N": nltk.corpus.wordnet.NOUN,
                "V": nltk.corpus.wordnet.VERB,
                "R": nltk.corpus.wordnet.ADV}

        return [(token, wordnet_tags.get(tag[0], nltk.corpus.wordnet.NOUN)) for (token, tag) in pos_tuples]

##############################################################################
## Noun Phrase Methods
##############################################################################

    def get_noun_phrases(
            self, 
            sentence, 
            noun_phrase_format=default_noun_phrase_format, 
            to_lowercase=False, 
            singularize=False
    ):
        """
        Derive all the noun phrases contained in a given sentence.
        get_noun_phrases(self, sentence, noun_phrase_format, to_lowercase, singularize)

        sentence: str
            The sentence to derive the noun phrases from.
        noun_phrase_format: str; optional (default = TextProcessor.noun_phrase_format)
            A string specifying how the noun phrases should be formatted/structured.
        to_lowercase: bool; optional (default=False)
            True if noun phrases should be converted to lowercase
        singularize: bool; optional (default = False)
            True if individual nouns within noun phrases should be singularized.

        Returns [str]; The list of noun phrases produced from the given sentence.
        Note: If no pos_tuples can be derived from the sentence, it will return np.nan. 
        This is useful for processing on a Pandas DataFrame without worrying about types.
        """
        
        # get all pos tuples
        pos_tuples = self.get_all_pos(sentence, to_lowercase=to_lowercase, lemmatize=singularize)

        # find the noun phrases based on the noun phrase format
        pos_tuples_noun_phrases = TextProcessor.__build_noun_phrases(pos_tuples, noun_phrase_format)
        
        if not isinstance(pos_tuples_noun_phrases, list):
            return np.nan
        
        return [token for (token, tag) in pos_tuples_noun_phrases if tag == 'NP']

    def __build_noun_phrases(
            pos_tuples, 
            noun_phrase_format=default_noun_phrase_format
    ):
        """
        <PRIVATE CLASS METHOD> Build the noun phrases by combining adjacent tuples that form a noun 
        phrase. Returns the list of pos_tuples with the noun phrases combined and labelled with the 
        tag 'NP'.
        __build_noun_phrases(pos_tuples, noun_phrase_format)

        pos_tuples: [(str, str)]
            The list of pos_tuples to derive noun phrases from.
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

##############################################################################
## Processing Methods
##############################################################################

    def process(
            self, 
            sentence, 
            to_lowercase=True, 
            preserve_noun_phrases=False, 
            remove_numbers=True, 
            custom_processing=lambda x:x
    ):
        """
        Tokenize, lemmatize and remove stopwords from a given sentence. Returns a list of tokens.
        Optionally convert tokens to lowercase, preserve noun phrases, remove numbers, and apply 
        custom processing. This method is intended for text pre-processing for machine learning 
        and other AI algorithms such as topic modelling, sentiment analysis, etc.
        process(self, sentence, to_lowercase, preserve_noun_phrases, remove_numbers, custom_processing)

        sentence: str
            The sentence to be processed.
        to_lowercase: bool; optional (default = True)
            True if tokens should be converted to lowercase.
        preserve_noun_phrases: bool; optional (default = False)
            True if noun phrases should be preserved in the list of tokens.
        remove_numbers: bool; optional (default = True)
            True if numbers/digits should be excluded from the list of tokens.
        custom_processing: str -> str; optional (default = lambda x: x)
            A function that takes in the sentence string and returns a string.
        
        Returns [str]; The list of lemmatized and non-stopword tokens from the given sentence.
        Note: If no pos_tuples can be derived from the sentence or if the sentence cannot be casted 
        to a string, it will return np.nan. This is useful for processing on a Pandas DataFrame 
        without worrying about types.
        """
        
        sentence = str(sentence)
        if sentence == 'nan':
            return np.nan 

        sentence = custom_processing(sentence)                                               # apply custom processing step
        pos_tuples = self.get_all_pos(sentence, to_lowercase=to_lowercase, lemmatize=True)   # get parts-of-speech

        # collect noun phrases if applicable
        if preserve_noun_phrases:
            pos_tuples = TextProcessor.__build_noun_phrases(pos_tuples)
            
        if not isinstance(pos_tuples, list):
            return np.nan
        
        # remove pos tags
        tokens = [token for (token, tag) in pos_tuples]

        # remove stopwords and punctuation
        filtered_tokens = []
        for token in tokens:
            if ( 
                token.lower() not in self.stopwords 
                and token not in TextProcessor.punctuation 
            ):
                filtered_tokens.append(token)

        if remove_numbers:
            filtered_tokens = list(filter(lambda token: re.search("[A-Za-z]", token) is not None, filtered_tokens))

        return filtered_tokens if len(filtered_tokens) > 0 else np.nan
    
