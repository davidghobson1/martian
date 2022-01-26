"""WordEmbedder

The WordEmbedder classes provide a simpler way to generate and visualize word embeddings
from unstructured text data.

This abstract class provides basic and unified functionality for general word embedding training, text processing,
embedding of words, but also embedding of sentences and embedding visualizations.

Filename: word_embedder.py
Maintainers: David Hobson, Saruggan Thiruchelvan
Last Updated: December 10, 2021
"""

import numpy as np
import pandas as pd

from text_processing import TextProcessor

import matplotlib.pyplot as plt
import random

from sklearn.manifold import TSNE                   # for visualizations

from abc import ABC, abstractmethod


class WordEmbedder(ABC):
    """WordEmbedder
    
    The WordEmbedder class is an abstract class providing basic functionality for word embeddings. 
    It makes it easy to build, train, and visualize embedding models, but also embed indivudal words and also 
    sentences. 
    
    Word embeddings create a learned representation of text by mapping words/sentences to a vector of floats; 
    words with similar meanings will have vectors with similar orientation (or cosine similarity). 
    This technique is used for many NLP tasks such as text summarization and training neural networks.

    Note: This class is abstract and can't be intiatated. Rather it serves as a template for subclasses like
    Word2VecEmbedder and PreTrainedEmbedder
    """
    
    def __init__(self, text_processor=TextProcessor().process, processing_kwargs={}):
        """
        Initialize a new WordEmbedder object.
        __init__(self, text_processor)

        text_processor: str -> [str]; optional (default = TextProcessor().process)
            A text processing function. This function must take a sentence str, and convert it
            into a list of tokens. It is recommended that this function also lemmatize, remove
            stopwords, and do any other useful routine to minimize the vocabulary size.
        processing_kwargs: dict; optional (default={})
            Optional additional arguments to pass to the processing function during sentence processing. 
            Must be keyword arguments (i.e. have format {'argument_name': argument_value})
        """
        
        self.text_processor = text_processor
        self.processing_kwargs = processing_kwargs
        self.embedding_model = None
        self.vector_size = None

    def get_embedding_size(self):
        """
        Returns the size of the embedding vectors.
        get_embedding_size(self)

        Returns: int; The size of the embedding vector
        """
        
        return self.vector_size

    def embed(self, word):
        """
        Returns the embedding vector for the given word. If the given word is not in the vocabulary
        of the embedding model, None is returned and the function prints a warning message.
        embed(self, word)

        word: str
            The word to be embedded
            
        Returns: [floats] or None; The embedding vector for the given word; None if the word is out
        of vocabulary.
        """
        
        if not isinstance(word, str):
            return np.nan

        try:
            return self.embedding_model[word]
        except(KeyError):
            print(word + " not found in vocabulary!")

    def embed_sentence(self, sentence):
        """
        Returns the embedding vector for a given sentence. This particular implementation returns 
        the average word embedding vector of all words in the sentence. 
        In particular, the sentence is processed (based on the processing function provided during 
        initialization) into tokens, each token in embedded, and the average of those vectors is returned.
        If the processed sentence does not contain any words in the embedding vocabulary, a message 
        is printed and the zero vector is returned.
        embed_sentence(self, sentence)

        word: str
            The sentence to be embedded
            
        Returns: [floats]; The embedding vector for the given sentence
        """
        
        if not isinstance(sentence, str):
            return np.nan

        processed_sentence = self.text_processor(sentence, **self.processing_kwargs)
        
        if not isinstance(processed_sentence, list):
            processed_sentence = []

        word_embeddings = []
        for word in processed_sentence:
            try:
                word_embeddings.append(self.embedding_model[word])
            except(KeyError):
                print(f"The word {word} was not found in the vocabulary!")
                continue

        # print message if there are no embedding vectors
        if len(word_embeddings) == 0:
          print("Found a sentence with no vocabulary words in it")
          print(f"\t{sentence}")
          word_embeddings = [np.zeros(self.vector_size)]

        return np.mean(word_embeddings, axis=0)
        
    @abstractmethod
    def initialize_model(self):
        """
        <ABSTRACT> Initializes the word embedding model. Must return a dict or dict-like
        object/mapping, which, when given a word, will give back the embedding vectore for that word.
        initialize_model(self)

        Returns: dict or dict-like
        """
        pass

    def _create_single_list(self, sentences):
        """  
        <PRIVATE> A helper function that works to generate a flattened (combined) list of sentences given a DataFrame, Series
        or list of sentences.
        _create_single_list(sentences)
        
        sentences : list, pd.Series, pd.DataFrame
            The list, Series, or DataFrame whose elements will be combined in a list.

        Returns: list; The combined list of all non-null elements
        
        Raises: ValueError, if sentences is not of type list, pd.Series, or pd.DataFrame
        """
        
        if isinstance(sentences, list):
            return [str(sentence) for sentence in sentences if str(sentence) != 'nan']
        elif isinstance(sentences, pd.Series):
            return list(sentences[sentences.notna()])
        elif isinstance(sentences, pd.DataFrame):
            return list(pd.concat([sentences[column][sentences[column].notna()] for column in sentences.columns]))
        else:
            raise ValueError('For word embeddings, data must either be in a list, series, of dataframe format')

    def visualize_embedding(self, words_to_plot=[], words_to_label=[]):
        """
        Prints a 2D representation of the embedding model, and labels the specified words.
        Words that are plotted close together are ones that the model has learned are similar in meaning.
        visualize_embedding(self, words_to_plot, words_to_label)

        words_to_plot: [str]; optional (default=[])
            The list of vocabulary words to plot
        words_to_label: [str], optional (default=[])
            The list of words to label on the plot. This must be a subset of words_to_plot.    
        
        Returns: None
        """
        
        x_vals, y_vals, labels = self._reduce_dimensions(words_to_reduce=words_to_plot, num_dimensions=2)
        WordEmbedder._plot_2D(x_vals, y_vals, labels, words_to_label)            
            
    def _reduce_dimensions(self, words_to_reduce=[], num_dimensions=2):
        """
        <PRIVATE> Reduces the embedding model to the specified number of dimensions. That is, the embedding vectors for the 
        specified vocabulary will be reduced to the given number of dimensions. 
        If the words_to_reduce argument is not specified, the entire embedding model vocabulary will be reduced. 
        Especially for pre-trained models, it is highly recommended that this list be provided since these models will
        have enormous vocabulary sizes, and the corresponding embedding reduction will be enormously slow.
        _reduce_dimensions(self, words_to_reduce, num_dimensions)

        words_to_reduce: [str], optional (default=[])
            The list of vocabulary words to reduce. If not-specified, the entire vocabulary is used.
        num_dimensions: int, optiona; (default = 2)
            The number of dimensions to reduce to.

        Returns: tuple of lists; The reduced vectors and their corresponding word
        The first list in the tuple will be the list of first coordinates for the words (i.e the x-values), 
        The second list will be the list of second coordinates (y-values),
        etc. (up to the number of dimensions).
        The last list will be the corresponding list of words
        """
     
        # extract the words & their vectors, as numpy arrays
        if words_to_reduce == []:
            vectors = np.asarray(self.embedding_model.vectors)
            labels = np.asarray(self.embedding_model.index_to_key)  # fixed-width numpy strings
        else:
            vectors = []
            labels = []
            for vocab_word in words_to_reduce:
                try:
                    vectors.append(np.asarray(self.embedding_model[vocab_word]))
                    labels.append(vocab_word)
                except(KeyError):
                    continue
                    
        tsne = TSNE(n_components=num_dimensions, random_state=0)               # reduce using t-SNE
        vectors = tsne.fit_transform(vectors)
        
        x = []
        for i in range(num_dimensions):
            x.append([v[i] for v in vectors])
        x.append(labels)
        
        return tuple(x)

    def _plot_2D(x_vals, y_vals, labels, words_to_label=[]):
        """
        Creates and displays a 2-dimension plot.
        _plot_2D(x_vals, y_vals, labels, labelled_words)
        
        x_vals: [floats]
            The list of x-values to be plotted
        y_vals: [floats]
            The list of y-values to be plotted
        labels : [str]
            The list of words corresponding to the x and y-values
        labelled_words: str or [str]: TYPE, optional (default='random')
           The list of words to label on the plot

        Returns: None
        """
        
        random.seed(0)

        plt.figure(figsize=(12, 12))
        plt.scatter(x_vals, y_vals)
     
        # show the chosen labelled words on the graph, or choose 25 random words
        if words_to_label == []:
            display_word_indices = np.random.choice(len(labels), size=np.min([25, len(labels)]), replace=False)
        else:
            display_word_indices = np.where([True if label in words_to_label else False for label in labels])[0]

        for i in display_word_indices:
            plt.annotate(labels[i], (x_vals[i], y_vals[i]))
        
        plt.show()