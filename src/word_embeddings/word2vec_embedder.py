"""Word2VecEmbedder

The Word2VecEmbedder is a WordEmbedder class that creates the embeding model using the Word2Vec algorithm.

Filename: word2vec_embedder.py
Maintainers: David Hobson, Saruggan Thiruchelvan
Last Updated: December 13, 2021
"""

import gensim
from gensim.models import Word2Vec

from word_embeddings import WordEmbedder
from text_processing import TextProcessor

class Word2VecEmbedder(WordEmbedder):
    """
    The Word2VecEmbedder class is a child class of the WordEmbedder class, that uses the Word2Vec algorithm to create the word embeddings.
    
    Read more about Gensim's pre-trained models: https://radimrehurek.com/gensim/models/word2vec.html#pretrained-models
    """

    def __init__(self, sentences, text_processor=TextProcessor().process, processing_kwargs={}, vector_size=100, min_count=1, window=5, workers=4):
        """
        Initialize a new Word2VecEmbedder object.
        __init__(self, sentences, text_processor, vector_size, min_count, window, workers)

        sentences: [str], DataFrame, or Series
            Sentences to train the word embedding on.
        text_processor: str -> [str]; optional (default = TextProcessor().process)
            A text processing function. This function must take a sentence str, and convert it
            into a list of tokens. It is recommended that this function also lemmatize, remove
            stopwords, and do any other useful routine to minimize the vocabulary size.
        processing_kwargs: dict; optional (default={})
            Optional additional arguments to pass to the processing function during sentence processing. 
            Must be keyword arguments (i.e. have format {'argument_name': argument_value})
        vector_size: int
            The number of dimensions of the embedding vectors; optional (default = 100)
        
        (See the Gensim Word2Vec documentation for more details on the remaining arguments)
            
        min_count: int
            The minimum number of occurences a word has to appear to be included in the embedding model; optional (default = 1).
        window: int
            The maximum distance (in number of words) between the current and predicted word within a sentence; optional (default = 5)
        workers: int
            The number of worker threads to train the model; optional (default = 4)
        """

        super().__init__(text_processor=text_processor, processing_kwargs=processing_kwargs)
        self.embedding_model = self.initialize_model(sentences, vector_size, min_count, window, workers)
        self.vector_size = vector_size

    def initialize_model(self, sentences, vector_size=100, min_count=1, window=5, workers=4):
        """
        Initializes and trains the Word2Vec embedding model
        initialize_model(self, sentence, vector_size, min_count, window, workers)
        
        sentences: [str], pd.DataFrame, or pr.Series
            The list of sentences to train the model on.
        vector_size: int
            The number of dimensions of the embedding vectors; optional (default = 100)
        min_count: int
            The minimum number of occurences a word has to appear to be included in the embedding model; optional (default = 1).
        window: int
            The maximum distance (in number of words) between the current and predicted word within a sentence; optional (default = 5)
        workers: int
            The number of worker threads to train the model; optional (default = 4)

        Returns: dict or dict-like
        """
        
        sentences = self._create_single_list(sentences)

        processed_sentences = []
        for sentence in sentences:
            processed_sentence = self.text_processor(sentence, **self.processing_kwargs)
            if not isinstance(processed_sentence, list):
                continue
            processed_sentences.append(processed_sentence)

        return Word2Vec(processed_sentences, window=window, vector_size=vector_size, min_count=min_count, workers=workers).wv