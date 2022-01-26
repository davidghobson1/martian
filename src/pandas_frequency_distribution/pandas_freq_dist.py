"""PandasFreqDist

This class is help to make finding frequency distributions from Pandas DataFrames or Series quick and easy, and
to provide methods to explore the resulting frequency distributions

Filename: pandas_freq_dist.py
Maintainers: David Hobson, Saruggan Thiruchelvan
Last Updated: December 10, 2021
"""

import numpy as np
import pandas as pd

from nltk import FreqDist

class PandasFreqDist:
    """PandasFreqDist
    
    This class computes frequency distributions from Pandas DataFrames or Series and provides methods to explore those distributions.
    
    Functionality includes:        
        - finding frequency distributions for multiple columns at once
        - automatically handling lists
        - getting top results for different columns
        - counting the number of occurrences in a column
        - handling groupings
    """
    
    def __init__(self, df):
        """
        Initialize a new PandasFreqDist object
        __init__(self, df)
        
        df: pd.DataFrame or pd.Series
            The data for which frequency distributions will be computed. A frequency distribution is created 
            for each column in the dataframe. The data within the columns or series can be of any type
        """
        
        if isinstance(df, pd.Series):                           # convert Series into DataFrames if necessary
            df = df.to_frame()   
        self.data = df
        self.fds = PandasFreqDist.compute_fds(df)

###############################################################################    
### Retrieving Top Result Methods
############################################################################### 

    def get_top_results(self, n, column, excluded_entries=[]):
        """
        Returns the top n most frequently occurring entries, along with their counts, for the given coloumn.
        The results are returned as two lists, with the entries and counts separate, which is convenient for plotting.
        If it's desired that the top entries and their counts be together in a tuple, use NLTK's most_common() method.
            ex. pfd.fds[column].most_common(n)
            The only downside with most_common() is that entries can't be excluded.
        get_top_results(self, n, column, excluded_entries)
        
        n: int
            The number of top results to return
        column: str
            The name of the column to be searched 
        excluded_entries: [object]; optional (default = [])
            Any entries in this list will be excluded from the top results returned. Note that n results will
            still be returned regardless.
            
        Returns: ([object], [int]); The top n results as well as their corresponding counts.
        Results are a tuple of two lists. The first is the top n results in descending
        order of frequency. The second contains the corresponding counts.
        """         

        return PandasFreqDist.get_top_results_fd(self.fds[column], n, excluded_entries)

    def get_top_results_by_group(self, n, column, groups, group_names=None, excluded_entries=[]):
        """
        Returns the top n most frequently occurring entries, along with their counts, for each group for the given column.
        The results are returned as a dict with keys being the group names (default simply numbers the groups), and the values
        being a 2-tuple of lists. The first list are the top n entries for the group in descending order, and the second list 
        are the corresponding counts for those entries.
        Entries in the excluded_entries argument will be excluded from the results.
        To get only the top entries without the counts, use the get_top_entries_by_group method.
        get_top_results_by_group(self, n, column, groups, group_names, excluded_entries)
        
        n: int
            The number of top results to return
        column: str
            The name of the column to be searched
        groups: [[int]]
             The list of the indices for each group, where each group indices are a list of ints. A list of lists of ints
        group_names: [str]; optional (default = None; this will simply number the groups) 
            The list of names for each group
        excluded_entries: [object]; optional (default = [])
            Any entries in this list will be excluded from the top results returned. Note that n results will
            still be returned however.
            
        Returns: {str: ([object], [ints])}; the top n most frequently occurring entries, with their counts, for each group for the given column.
        The dict has the group names as keys, and the values being a 2-tuple of lists. 
        The first is the top n most frequently occurring objects in descending order of frequency. The second contains the corresponding counts.
        """      
        
        if group_names is None:                                       # create default group names if necessary
            group_names = [str(i) for i in range(len(groups))]
            
        fds = self.get_fds_by_group(column, groups)                                                                             # get the frequency distributions for each group
        return {group_names[i]: PandasFreqDist.get_top_results_fd(fds[i], n, excluded_entries) for i in range(len(groups))}     # count the occurences for each group


    def get_top_entries(self, n, column, excluded_entries=[]):
        """
        Returns the top n most frequently occurring entries, however not including their counts, for the given coloumn.
        If the counts are also desired, use the get_top_results method instead.
        get_top_results(self, n, column, excluded_entries)
        
        n: int
            The number of top results to return
        column: str
            The name of the column to be searched 
        excluded_entries: [object]; optional (default = [])
            Any entries in this list will be excluded from the top results returned. Note that n results will
            still be returned however.
            
        Returns: [object]; The top n most frequently occuring entries in descending order.
        """ 
        
        return self.get_top_results(n, column, excluded_entries)[0]

    def get_top_entries_by_group(self, n, column, groups, group_names=None, excluded_entries=[]):
        """
        Returns the top n most frequently occurring entries for each group for the given column.
        The results are returned as a dict with keys being the group names (default simply numbers the groups), and the values
        being the list of the top n entries for the group in descending order.
        Entries in the excluded_entries argument will be excluded from the results.
        To get both the top entries and the corresponding counts, use the get_top_results_by_group method.
        get_top_entries_by_group(self, n, column, groups, group_names, excluded_entries)
        
        n: int
            The number of top results to return
        column: str
            The name of the column to be searched
        groups: [[int]]
             The list of the indices for each group, where each group indices are a list of ints. A list of lists of ints
        group_names: [str]; optional (default = None; this will simply number the groups) 
            The list of names for each group
        excluded_entries: [object]; optional (default = [])
            Any entries in this list will be excluded from the top results returned. Note that n results will
            still be returned however.
            
        Returns: {str: [object]}; The top n most frequently occurring entries for each group.
        The dict has the group names as keys, and the values being the lists of the top n most 
        frequently occurring objects in descending order of frequency.
        """     
        
        if group_names is None:                                       # create default group names if necessary
            group_names = [str(i) for i in range(len(groups))]
        
        group_results = self.get_top_results_by_group(n, column, groups, group_names, excluded_entries)
        return {group_name: group_results[group_name][0] for group_name in group_names}

    def get_top_results_fd(fd, n, excluded_entries=[]):
        """
        <CLASS METHOD> Returns the top n most frequently occurring entries, along with their counts, for the given NLTK frequency distribution.
        The results are returned as two lists, with the entries and counts separate, which is convenient for plotting.
        If only the top entries are needed, and not their counts, use the get_top_entries method.
        If it's desired that the top entries and their counts be together as a tuple, use NLTK's most_common() method instead.
            ex. fd.most_common(n)
        The only downside here is that entries can't be excluded.
        get_top_results_fd(fd, n, column, excluded_entries)
        
        fd: ntlk.FreqDist object
            The frequency distribution to search
        n: int
            The number of top results to return
        column: str
            The name of the column to be searched 
        excluded_entries: [object]; optional (default = [])
            Any entries in this list will be excluded from the top results returned. Note that n results will
            still be returned however.
            
        Returns: ([object], [int]); The top n most frequently occurring entries and their counts.
        A tuple of two lists. The first is the list of top n results in descending
        order of frequency. The second list contains the corresponding counts for the objects in the first.
        """  
        
        entries, counts = [], []
        curr_num_results = 0
        for item, count in fd.most_common():
            if curr_num_results == n:
                break
            elif item in excluded_entries:
                continue
            entries.append(item)
            counts.append(count)
            curr_num_results += 1
        return entries, counts 

###############################################################################    
### Counting Occurrences Methods   
############################################################################### 
  
    def count_occurrences(self, column, entries):
        """
        Counts the number of times the given entries occur within the given column
        count_occurrences(self, column, entries)
        
        column: str
            The name of the column to be searched 
        entries: [object]
            The list of entries to be searched for
            
        Returns: [int]; The number of times each entry occurs in the column
        """            

        return PandasFreqDist.count_occurrences_from_fd(self.fds[column], entries)

    def count_occurrences_by_group(self, column, entries, groups):
        """
        Counts the number of times the given entries occur for each group within the given column.
        count_occurrences_by_group(self, column, entries, groups)
        
        column: str
            The name of the column to be searched 
        entries: [object]
            The list of entries to be searched for
        groups: [[int]]
             The list of the indices for each group, where each group indices are a list of ints. A list of lists of ints
            
        Returns: {object: [int]}; The number of times the given entries occur for each group
        A dict with the entries as keys, and values being the list of counts of that entry 
        for each group. The counts correspond to the order of the groups in the groups argument.
        """  
        
        fds = self.get_fds_by_group(column, groups)                                                   # get the frequency distributions for each group
        return {entry:[fds[i][entry] for i in range(len(groups))] for entry in entries}               # count the occurences for each group
     
    def count_occurrences_from_fd(fd, entries):
        """
        <CLASS METHOD> Counts the number of times the entries occur within the given NLTK frequency distribution
        count_occurences_from_fd(fd, entries)

        fd: nltk.FreqDist object
            The frequency distribution to search
        entries: [object]
            The list of entries to search for
            
        Returns: [int]; The number of times each entry occurs in the frequency distribution
        """
        
        return [fd[entry] for entry in entries]
        
###############################################################################    
### Percent Usage Methods 
############################################################################### 

    def get_percent_usage_by_group(self, column, entries, groups):
        """
        Returns the percentage of (non-null) series elements that contain each of the given entries for each of the given groups.
        Note that if an entry occurs more than once with a given series element, it is only counted once.
        get_percent_uage_by_group(self, column, entries, groups)
        
        column: str
            The name of the column
        entries: [object]
            The list of entries to compute percentages for
        groups: [[int]]
            The list of the indices for each group, where each group indices are a list of ints. A list of lists of ints
            
        Returns: {object: [float]}; The percentage of (non-null) series elements that contain each of the given entries for each 
        of the given groups.
        A dict with the entries as keys, and values as the list representing the percentages that entry occurs for each group. 
        The order of percentages is the same as the order of the groups in the groups argument.
        """          
        
        df = self.data[column]
        
        # get percentages for each group (array shape: group x percentages) 
        percentages_by_group = np.array([PandasFreqDist.get_percent_usage_series(df.loc[group], entries) for group in groups])
        
        # flip to get percentages from each group (array shape: entry x percentage for each group)
        percentages_by_entry = percentages_by_group.T
        
        return {entries[i]: percentages_by_entry[i] for i in range(len(entries))}
    
    def get_percent_usage_series(series, entries):
        """
        <CLASS METHOD> Returns the percentage of (non-null) series elements that contain each of the given entries.
        Note that if an entry occurs more than once with a given series element, it is only counted once.
        get_percent_uage_by_group(series, entries)
        
        series: pd.Series
            The series to be searched. 
        entries: [object]
            The entries which will be searched for in the series.

        Returns: np.array(float); The percentage of (non-null )series elements that contain each entry. The percentages are
        in the same order as the given entries in the entries argument.
        """
        
        series = series[~series.isnull()]
        n = len(series)
        percentages = []
        for entry in entries:    
            num_encounters = 0
            for series_item in series:
                try:
                    if entry in series_item:
                        num_encounters += 1
                except(TypeError):
                    continue
            percentages.append(np.round(num_encounters/n, 3)) 
        return np.array(percentages)*100
   
###############################################################################    
### General Purpose Methods 
############################################################################### 
    
    def get_fds_by_group(self, column, groups):
        """
        Get the frequency distributions for each group for the given column.
        get_fds_by_group(self, column, groups)
        
        column: str
            The name of the column
        groups: [[int]]
            The list of the indices for each group, where each group indices are a list of ints. A list of lists of ints
            
        Returns: [nltk.FreqDist]; The list of frequency distributions for each group for the given column.
        """          
        
        return [FreqDist(PandasFreqDist.create_single_list(self.data[column].loc[group])) for group in groups]

    def compute_fds(df):
        """
        <CLASS METHOD> Computes the frequency distributions for all of the columns in the given
        dataframe.
        create_single_list(series)
        
        df: pd.DataFrame or pd.Series
            The df from which frequency distributions are computed
            
        Returns: dict; The frequency distributions.
        Keys are the column names, and values are the nltk.FreqDist objects
        """        
        
        fds = dict()
        for col, series in df.items():
            fds[col] = FreqDist(PandasFreqDist.create_single_list(series))  
        return fds
        
    def create_single_list(series):
        """
        <CLASS METHOD> Creates a list of all of the entries in a series. This includes
        unpacking elements in the series that may be lists themselves. In this case, all
        elements, whether single entries or within a list in the series, are all placed 
        into one list. 
        create_single_list(series)
        
        series: pd.Series
            The series to build a list from
            
        Returns: [object]; The list of all of the elements (including individual list items) from the series.
        """  
    
        series = series[~series.isnull()]
        
        all_entries = []
        for entry in series:
            if not isinstance(entry, list):                
                all_entries.append(entry)
            else:
                for item in entry:
                    all_entries.append(item)
        
        return all_entries
          
            