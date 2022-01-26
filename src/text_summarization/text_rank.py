"""TextRanker

The TextRanker class makes it easy to apply the TextRank algorithm.

This algorithm is a text summarization algorithm that is based on Google's PageRank algorithm.

Filename: text_rank.py
Maintainers: David Hobson, Saruggan Thiruchelvan
Last Updated: December 13, 2021
"""

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

from word_embeddings import WordEmbedder

class TextRanker:   
    """
    The TextRanker class automates and simplifies applying the TextRank text summarization algorithm. This algorithm
    is based on and inspired by Google's PageRank algorithm.
    
    At a high level, this algorithm 'summarizes' text by ranking each sentence or entry based on how semantically-similar 
    it is to every other sentence/entry. In this way, the top-ranked sentence is the one that is the most similar to all 
    of the other sentences, and therefore provides the 'best summary' of the entire text. 

    Use of this class only requires an embedding model and the list of sentences to apply the algorithm. 
    
    At this stage, only word embeddings that descend from the WordEmbedder class are accepted for word embeddings.
    This is because TextRank requires embeddings of sentences which are already implemented into the WordEmbedder class.
    
    The class makes use of the PageRank algorithm by the NetworkX Python library.
    
    For more details on this algorithm, see the tutorial notebook on this repo or other online resources.    
    """
    
    def __init__(self, embedding_model):
        """
        Initialize a new TextRanker object.
        __init__(self, embedding_model)
        
        embedding_model: WordEmbedder object
            The embedding model to embed the sentences/entries before application of the TextRank algorithm.
            This must be capable of embedding sentences.
        """
    
        self.embedding_model = embedding_model

    def text_rank(self, sentences, alpha=0.85, max_iter=100, tol=1e-6):
        """
        Ranks the given sentences according to the TextRank algorithm. 
        text_rank(self, sentences, alpha, max_iter, tol)
        
        sentences: [str] or pd.Series
            The sentences to be ranked
        alpha: float; optional (default=0.85)
            The damping parameter for applying the PageRank algorithm.
        max_iter: int; optional (default=100)
            The maximum number of iterations for the PageRank algorithm.
        tol: float; optional (default = 1e-6)
            The error tolerance used to check for convergence of PageRank.
        
        For more details of the latter 3 arguments, see the NetworkX documentation of the PageRank algorithm:
        https://networkx.org/documentation/networkx-1.2/reference/generated/networkx.pagerank.html
        
        Returns: pd.DataFrame; 
        The results from TextRank. The sentences in the DataFrame are already sorted by score and are 
        listed in decreasing order of importance. 
        The DatFrame has only two columns: 'Score' and 'Sentence', which give the score of each sentence as determined
        by TextRank, and the corresponding sentence. 
        As a note: all scores across all sentences should sum to 1.
        """
        # convert to Series for easier indexing
        if not isinstance(sentences, pd.Series):
            sentences = pd.Series(sentences)

        # embed the sentences
        embedded_sentences = sentences.apply(lambda x: self.embedding_model.embed_sentence(x))

        # # filter out null sentences
        not_null_indices = embedded_sentences[embedded_sentences.notnull()].index
        embeddings = embedded_sentences.loc[not_null_indices]
        sentences = sentences.loc[not_null_indices]

        # compute the similarity matrix
        embeddings = [np.array(i) for i in embeddings]   # convert series to numpy array since cosine similarity needs an numpy array of size (# datapoints x # features)
        similarity_matrix = cosine_similarity(embeddings, embeddings)
        for i in range(len(similarity_matrix)):
            similarity_matrix[i][i] = 0

        # create the graph (similarity matrix is interpreted as an adjacency matrix)
        nx_graph = nx.from_numpy_array(similarity_matrix)

        # apply PageRank
        print("Applying PageRank...")
        scores = nx.pagerank(nx_graph, alpha=alpha, max_iter=max_iter, tol=tol)
        print("Done")

        # associate the node's score with the sentence
        results = [ (scores[node_number], sentences.iloc[node_number]) for node_number in scores.keys() ]

        # return the results in a sorted DataFrame
        return pd.DataFrame(data=results, columns=['Score', 'Sentence'], index=not_null_indices).sort_values(by=['Score'], ascending=False)
