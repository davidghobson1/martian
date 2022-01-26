"""PreTrainedEmbedder

The PreTrainedEmbedder is a WordEmbedder class that acts as a wrapper for using
one of Gensim's pre-trained word embedding models.

Filename: pretrained_embedder.py
Maintainers: David Hobson, Saruggan Thiruchelvan
Last Updated: December 13, 2021
"""

import gensim
from gensim.downloader import load

from word_embeddings import WordEmbedder
from text_processing import TextProcessor

class PreTrainedEmbedder(WordEmbedder):

    def __init__(self, text_processor = TextProcessor().process, processing_kwargs={}, gensim_model='word2vec-google-news-300'):
        """
        Initialize a new PreTrainedEmbedder object.
        __init__(self, text_processor, gensim_model)
        
        text_processor: str -> [str]; optional (default = TextProcessor().process)
            A text processing function. This function must take a sentence str, and convert it
            into a list of tokens. It is recommended that this function also lemmatize, remove
            stopwords, and do any other useful routine to minimize the vocabulary size.
        processing_kwargs: dict; optional (default={})
            Optional additional arguments to pass to the processing function during sentence processing. 
            Must be keyword arguments (i.e. have format {'argument_name': argument_value})
        gensim_model: str; optional (default = 'word2vec-google-news-300')
            The name of the Gensim embedding model to use. The default is a model with embedding vectors of length 300
            that was trained using Word2Vec on articles from Google News.
        """

        super().__init__(text_processor=text_processor, processing_kwargs=processing_kwargs)

        self.embedding_model = self.initialize_model(gensim_model)
        self.vector_size = len(self.embed("tree"))
        self.model_name = gensim_model

    def initialize_model(self, model_name):
        """
        Loads the specified Gensim embedding model.
        initialize_model(self, model_name)
        
        model_name: str;
            The name of the Gensim model to load.
            
        Returns: dict (or dict-like)
        """
        return load(model_name)
