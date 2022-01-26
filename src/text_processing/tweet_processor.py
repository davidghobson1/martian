"""TweetProcessor

This class adds additional functionality to the TextProcessor class for processing tweets.

Filename: tweet_processor.py
Maintainers: David Hobson, Saruggan Thiruchelvan
Last Updated: December 7, 2021
"""

import numpy as np

import nltk
from text_processing.text_processor import TextProcessor

class TweetProcessor(TextProcessor):
    """TweetProcessor
    
    This class is an extension of the TextProcessor class, and is specifically designed to process tweets. 
    
    Additional features include the ability to identify Twitter hashtags, Twitter handles, and links. However all
    processing is tailored to identify and preserve specific Twitter information. This includes small things like 
    preserving case for things like Twitter account names, since these are case-sensitive.
    """
    
    def __init__(
        self, 
        lemmatizer = nltk.WordNetLemmatizer().lemmatize, 
        stopwords = nltk.corpus.stopwords.words('english')
    ):
        """
        Initialize a new TweetProcessor object
        This tokenizer for this object uses NLTK's TweetTokenizer class
        __init__(self, lemmatizer, stopwords)
        
        lemmatizer: str -> str; optional (default = WordNetLemmatizer().lemmatize)
            Function that takes in a token string and returns the lemmatized token string.
        stopwords: [str]; optional (default = nltk.corpus.stopwords.words('english))
            The list of words to be excluded and ignored during processing.
        """
        
        super().__init__(tokenizer=nltk.TweetTokenizer().tokenize, lemmatizer=lemmatizer, stopwords=stopwords)

##############################################################################
## Twitter-specific Methods
##############################################################################
  
    def get_hashtags(self, tweet):
        """
        Get all the Twitter hashtags from a given tweet string.
        get_hashtags(self, tweet)

        tweet: str
            A tweet to get handles from.

        Returns [str]; The list of hashtags derived from the given tweet.
        """

        tokens = self.tokenize(tweet)
        return [token for token in tokens if TweetProcessor.is_hashtag(token)]

    def get_handles(self, tweet):
        """
        Get all the Twitter handles from a given tweet string.
        get_handles(self, tweet)

        tweet: str
            A tweet to get handles from.

        Returns [str]; The list of handles derived from the given tweet.
        """

        tokens = self.tokenize(tweet)
        return [token for token in tokens if TweetProcessor.is_handle(token)]        

    def is_twitter_element(token):
        """
        Check if a token string is a hashtag, Twitter handle, or link
        is_twitter_element(token)

        token: str
            A token to be checked.

        Returns boolean; True if token represents a hashtag, Twitter handle, link.
        """
        
        if (
            TweetProcessor.is_hashtag(token)
            or TweetProcessor.is_handle(token)
            or TweetProcessor.is_link(token)
        ):
            return True
        else:
            return False
  

    def is_hashtag(token):
        """
        Check if a token string is a Twitter hashtag.
        is_hashtag(token)

        token: str
            A token to be checked.

        Returns boolean; True if token represents a hashtag.
        """
        return token[0] == '#'

    def is_handle(token):
        """
        Check if a token string is a Twitter handle.
        is_handle(token)

        token: str
            A token to be checked.

        Returns boolean; True if token represents a handle.
        """
        return token[0] == '@'


    def is_link(token):
        """
        Check if a token string is a link.
        is_link(token)

        token: str
            A token to be checked.

        Returns boolean; True if token represents a link.
        """
        return False if len(token) < 4 else token[:4] == 'http'

##############################################################################
## Tokenization Methods
##############################################################################

    def tokenize(self, tweet, to_lowercase=False):
        """
        Get the tokens from a given tweet. Hashtags, handles, and links will also be returned.
        tokenize(self, tweet, to_lowercase=False)

        tweet: str
            A tweet to be tokenized.
        to_lowercase: bool; optional (default = False)
            True if tokens should be converted to lowercase.

        Returns [str]; The list of tokens derived from the given tweet.
        """
        
        original_tokens = super().tokenize(tweet, to_lowercase=to_lowercase, lemmatize=False)

        # split contractions since the Tweet Tokenizer doesn't do that
        tokens = []
        for token in original_tokens:
            if TweetProcessor.is_twitter_element(token):
                tokens.append(token)
            else:
                split_token = nltk.word_tokenize(token)
                for word in split_token:
                    tokens.append(word)
        
        return tokens
        
    def tokens_to_lowercase(self, tokens):
        """
        Convert each token in list of tokens to lowercase. Since Twitter account names and 
        links are case-sensitive, these are not converted to lowercase.
        tokens_to_lowercase(self, tokens)

        tokens: [str]
            List of tokens to be converted to lowercase.

        Returns [str]; The given list of tokens converted to lowercase.
        """
        if not isinstance(tokens, list):
            return np.nan
      
        return [token.lower() if not (TweetProcessor.is_handle(token) or TweetProcessor.is_link(token)) else token for token in tokens]

##############################################################################
## Parts-of-Speech (POS) Methods
##############################################################################

    def get_all_pos(
        self, 
        tokens, 
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
        
        pos_tuples = super().get_all_pos(tokens, to_lowercase=to_lowercase, lemmatize=lemmatize)
        return self.__correct_pos_mistakes(pos_tuples)
    
    def __correct_pos_mistakes(self, pos_tuples):
        """
        <PRIVATE> Attempt to correct the parts-of-speech tagging mistakes
        Sometimes hashtags, handles, and links are tagged as verbs. This changes 
        those to be tagged as nouns by default instead.
        __correct_pos_mistakes(self, pos_tuples)

        pos_tuples: [(str, str)]
            List of pos_tuples to be corrected.

        Returns [(str, str)]; The given list of pos_tuples with corrections made for hashtags, handles, and links.
        """  
    
        for i in range(len(pos_tuples)):
            token, tag = pos_tuples[i]
            if TweetProcessor.is_twitter_element(token) and ('VB' in tag):
                pos_tuples[i] = (token, 'NN')  
        return pos_tuples
 
##############################################################################
## Lemmatization Methods
##############################################################################
    
    def lemmatize_pos_tuples(self, pos_tuples):
        """
        Lemmatize the token part of each tuple in the given list of pos_tuples.
        Tweet-specific elements like hashtags, handles, etc. are not lemmatized
        lemmatize_pos_tuples(self, pos_tuples)

        pos_tuples: [(str, str)]
            The list of pos_tuples to be lemmatized.
        
        Returns [(str, str)]; The list of pos_tuples with the tokens lemmatized.
        """
        
        pos_tuples_wordnet = TweetProcessor._TextProcessor__format_pos_tuples_to_wordnet(pos_tuples)
        
        lemmatized_pos_tokens = [self.lemmatizer(token, pos=tag) if not TweetProcessor.is_twitter_element(token) else token for (token, tag) in pos_tuples_wordnet]
        original_pos_tags = [tag for (token, tag) in pos_tuples]

        # match each token with their original pos-tag
        return [(lemmatized_pos_tokens[i], original_pos_tags[i]) for i in range(len(pos_tuples))]
      
##############################################################################
## Processing Methods
##############################################################################
    
    def process(
            self, 
            sentence, 
            to_lowercase=True, 
            preserve_noun_phrases=False, 
            remove_numbers=True, 
            ignore_hashtags=True, 
            ignore_handles=True, 
            ignore_links=True, 
            custom_processing=lambda x:x
    ):
        """
        Tokenize, lemmatize and remove stopwords from a given tweet. Returns a list of tokens.
        Optional configurations include converting tokens to lowercase, preserving noun phrases, removing numbers 
        and ignore Twitter elements such as hashtags, handles, and links, and as applying custom processing.
        process(self, sentence, to_lowercase, preserve_noun_phrases, remove_numbers, custom_processing)

        sentence: str
            The sentence to be processed and derive tokens from.
        to_lowercase: bool; optional (default = True)
            True if tokens should be converted to lowercase.
        preserve_noun_phrases: bool; optional (default = False)
            True if noun phrases should be preserved in the list of tokens.
        remove_numbers: bool; optional (default = True)
            True if numbers/digits should be excluded from the list of tokens.
        ignore_hashtags: bool; optional (default = True)
            True if hashtags should be excluded from the list of tokens.
        ignore_handles: bool; optional (default = True)
            True if handles should be excluded from the list of tokens.
        ignore_links: bool; optional (default = True)
            True if links should be excluded from the list of tokens.
        custom_processing: str -> str; optional (default = lambda x: x)
            A function that takes in the sentence string and returns a string.
        
        Returns [str]; The list of tokens produced from the given tweet.
        Note: If no pos_tuples can be derived from the tweet or if the tweet cannot be casted to,
        a string, it will return np.nan. This is useful for processing on a Pandas DataFrame without
        worrying about types.
        """
        tokens = super().process(sentence, to_lowercase=to_lowercase, preserve_noun_phrases=preserve_noun_phrases, remove_numbers=remove_numbers, custom_processing=custom_processing)
        if ignore_hashtags:
            tokens = [token for token in tokens if not TweetProcessor.is_hashtag(token)]
        if ignore_handles:
            tokens = [token for token in tokens if not TweetProcessor.is_handle(token)]
        if ignore_links:
            tokens = [token for token in tokens if not TweetProcessor.is_link(token)]
        return tokens