"""PandasWordFreqDist

This class adds additional functionality to the PandasFreqDist class specifically designed for handling text data.

Filename: pandas_word_freq_dist.py
Maintainers: David Hobson, Saruggan Thiruchelvan
Last Updated: December 10, 2021
"""

import numpy as np
import pandas as pd

import re

from pandas_frequency_distribution.pandas_freq_dist import PandasFreqDist
    
class PandasWordFreqDist(PandasFreqDist):    
    """PandasWordFreqDist
    
    This class is an extension of the PandasFreqDist class and is designed to provide addition functionality to handle text data
    
    Additional functionality includes:        
        - handling synonyms
        - doing partial matching
    """    
    
    def __init__(self, df, synonyms={}):
        """
        Initialize a new PandasWordFreqDist object
        __init__(self, df, synonyms)
        
        df: pd.DataFrame or pd.Series
            The data for which frequency distributions will be computed. A frequency distribution is created 
            for each column in the dataframe.
        synonyms: {str: [str]}; optional (default = {})
            An optional dictionary of synonyms. The keys of the dict are all the standard words, and the values correspond to the 
            list of words that are equivalent to that key.
        """

        if isinstance(df, pd.Series):
            df = df.to_frame()
    
        self.original_synonyms = synonyms
        self.synonyms = PandasWordFreqDist._PandasWordFreqDist__create_direct_synonym_dict(synonyms)
    
        df = self.__replace_synonyms(df)
        super().__init__(df)
        
###############################################################################    
### Handling Synonym Methods    
###############################################################################
        
    def get_synonyms(self):
        """
        Get the dictionary of sysnonyms
        get_synonyms(self)

        Returns: {str: [str]}
            The dictionary of synonyms. The keys are all the standard words, and the values correspond to the 
            list of words that are equivalent to that key.
        """
        
        return self.original_synonyms
    
    def __create_direct_synonym_dict(synonym_dict):
        """
        <PRIVATE CLASS METHOD> Converts a synonyms dictionary from a format where a standard word is associated with 
        a list of words that are all synonymous to it (i.e. a dictionary of lists), to a dictionary where a word is 
        mapped directly to its standard synonym (i.e. a dictionary of words).
        This is stored internally by the PandasWordFreqDist object and is used to boost the efficiency of synonym searches.
        create_direct_synonym_dict(synonym_dict)
        
        synonym_dict: {str: [str]}
            A dictionary of standard words and the corresponding list of words synonymous to them

        Returns: {str: str}; The dictionary of words mapped to their standard synonym
        """
        
        direct_synonym_dict = {}
        for standard_word in synonym_dict:
            synonyms_list = synonym_dict[standard_word]
            for synonym in synonyms_list:
                direct_synonym_dict[synonym] = standard_word
        return direct_synonym_dict
    
    def __replace_synonyms(self, df):
        """
        <PRIVATE> Replace all the words in the dataframe with their standard synonym.
        Helper method for the PandasWordFreqDist constructor. 
        __replace_synonyms(self, df)
        
        df: pd.DataFrame
            The dataframe within which synonyms will be replaced. 

        Returns: pd.DataFrame; The dataframe with the synonyms replaced and standardized
        """
        
        synonyms = self.synonyms
        for name, series in df.items():   
            non_null_inds = series[~series.isnull()].index
            for index in non_null_inds:                
                curr_entry = df[name].loc[index]
                if not isinstance(curr_entry, list):                                # replace all words with synonyms for both single entries and lists
                    df[name].loc[index] = synonyms.get(curr_entry, curr_entry)
                else:   
                    list_ = curr_entry
                    df[name].loc[index] = [synonyms.get(item, item) for item in list_]
        return df

###############################################################################    
### Counting Occurrences Methods
###############################################################################    

    def count_occurrences(self, column, entries, partial_match=False):
        """
        Counts the number of times the given entries occur within the given column. If partial_match is True, words that partially
        match the entries will also be included in the count
        count_occurrences(self, column, entries, partial_match)
        
        column: str
            The name of the column to be searched 
        entries: [object]
            The list of entries to be searched for
        partial_match: bool; optional (default = False)
            Indicates whether words that partially match a given entry should be included in the count          
            
        Returns: [int]; The number of times each entry occurs in the column
        """ 
        
        if not partial_match:
            return super().count_occurrences(column, entries)
        else:
            partial_matches = self.get_partial_matches(column, entries)
            return [np.sum(list(partial_matches[key].values())) for key in partial_matches]
        
    
    def get_partial_matches(self, column, entries):
        """
        Finds all partial or full matches of the given entries within the given columns. Returns a dict containing 
        all the entries paired with another dictionary containing all the matchs and their corresponding counts.
        get_partial_matches(self, column, entries)
        
        column: str
            The name of the column to search in
        entries: [str]
            The entries to be searched for

        Returns: {str: {str: int}}; The dict representing all the partial matches. The entry values are the keys,
        and their values are a dictionary with all the partial or full matches along with the number of times they
        occur.
        """
        
        series = self.data[column][~self.data[column].isnull()]
        
        results = dict()
        for entry in entries:
            partial_matches = dict()
            for series_element in series:                                                # iterate over all the series entries to find matches
                if not isinstance(series_element, list):
                    num_occurrences = len(re.findall(entry, series_element))
                    if num_occurrences > 0:
                        if series_element in partial_matches:
                            partial_matches[series_element] += num_occurrences
                        else:
                            partial_matches[series_element] = num_occurrences
                else:
                    list_ = series_element
                    for item in list_:   
                        num_occurrences = len(re.findall(entry, item))
                        if num_occurrences > 0:
                            if item in partial_matches:
                                partial_matches[item] += num_occurrences
                            else:
                                partial_matches[item] = num_occurrences
            results[entry] = partial_matches
        
        return results
        