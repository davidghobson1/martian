"""Azure API Helper

This module simplifies and automates calls to the Azure API. 

This includes making batches, as well as data formating when making calls to an Azure API service.

At this stage, it can only make calls to the Translation API, however calls
to other APIs can be implemented moving forward.
"""

import numpy as np
import pandas as pd

import nltk                                  # needed to parse sentences if individual entries are too big for Azure to handle
from nltk import sent_tokenize
nltk.download('punkt')

import requests
import json
from collections import defaultdict
from time import sleep

# the maximum number characters and maximum array size allowed by the Azure Translation API.
# these are found here: https://docs.microsoft.com/en-us/azure/cognitive-services/translator/request-limits
MAX_CHARS = 10000
MAX_BATCH_SIZE = 100

# get the supported language information from the Azure Translation API
supported_languages_info = json.loads(requests.get('https://api.cognitive.microsofttranslator.com/languages?api-version=3.0').text)
translation_supported_lang_info = supported_languages_info['translation']

# get supported language codes and names
supported_lang_codes = set(translation_supported_lang_info.keys())
supported_lang_names = defaultdict(lambda : 'Unknown')
for lang_code in translation_supported_lang_info:
  lang_name = translation_supported_lang_info[lang_code]['name'].lower()
  supported_lang_names[lang_name] = lang_code

# the headers to use when making a request to the API
api_call_headers = {}

def set_api_headers(subscription_key, subscription_region):
  """
  Set the subscription key and subscription region for API requests made by 
  this module. 
  set_api_headers(subscription_key, subscription_region)
  
  subscription_key: str
    The subscription key for the Azure API subscription
  subscription_region: str
    The subscription region for the Azure API subscription
  """
  api_call_headers['Ocp-Apim-Subscription-Key'] = subscription_key
  api_call_headers['Ocp-Apim-Subscription-Region'] = subscription_region
  api_call_headers['Content-Type'] = 'application/json'

def translate(df, language='en', columns=None, batch_size=50, batch_char_limit=10000, wait_time=0, display_progress=False, print_errors_verbose=False):
  """
  Translates the entries in the given DataFrame, Series, or list, to the specified language. Returns the translated data.
  Null values are left as null in the translated data.
  If applicable, specific columns can be specified to only translate certain columns.
  If errors occur during translation, the indices corresponding to those errors will be printed at the end. No indices printed
  means that no errors occurred.
  For those errors, the values in the DataFrame, Series, or list will be the error object from the Azure Translation API.
  These are returned as dictionaries, which can be used to distiguish between successful and unsuccessful translations.
  For large datasets or datasets with long entries, it is recommended that the wait_time argument be non-zero to avoid
  throttling errors with the API.
  For more information about errors using the Translation API, see the following link:
  https://docs.microsoft.com/en-us/azure/cognitive-services/translator/reference/v3-0-reference#errors
  
  translate(df, language, columns, batch_size, batch_char_limit, wait_time, display_progress, print_errors_verbose)
    
  df : pd.DataFrame, pd.Series, or list
    The data to be translated.
  language: str; optional (default='en'; corresponds to English)
    The name or language code of the language to translate to.
  columns: list; optional (default=None; indicates all columns to be translated)
    The specific columns to be translated.
  batch_size: int; optional (default=50)
    The maximum number of entries which will be sent to the translation API at once.  
  batch_char_limit: int; optional (default = 10,000)
    The maximum number of characters which will be sent to the translation API at once.
  wait_time: int; optional (default=0)
    The amount of time, in seconds, between sending requests to the Azure translation API.
    It is recommended this be non-zero if the dataset is quite large, or entries are long. With an S1 subscription, Azure can only 
    process 11,111 characters per second, so adjusting this wait time is sometimes necessary to ensure the API doesn't generate 
    errors from being overloaded.
    One such error is the throttling error. If this error is encountered, increase this value and the problem should go away.
  display_progress: bool; optional (default=False)
    True if messages should be printed to display the number of entries currently translated.
  print_errors_verbose: bool; optional (default=False)
    True if more detailed error messages should be printed, when encountered.
    
  Returns: pd.DataFrame, pd.Series, or list (same type as df argument); The translated data.
  """
    
  # get language code and ensure it is supported by the Azure API
  language_code = language if language in supported_lang_codes else supported_lang_names[language.lower()]
  if language_code == 'Unknown':
    display_language_error(language)
    return
  
  # make sure batch size and character limit adhere to Azure API limits
  batch_violation = True if batch_size > MAX_BATCH_SIZE or batch_char_limit > MAX_CHARS else False
  if batch_violation:
    display_batch_violation_error(batch_size, batch_char_limit)
    return
  
  # routines for different data types
  if isinstance(df, list):
    df = pd.Series(df)
    return list(_translate_series(df, language_code, batch_size, batch_char_limit, wait_time, display_progress, print_errors_verbose).values)
  if isinstance(df, pd.Series):
    return _translate_series(df, language_code, batch_size, batch_char_limit, wait_time, display_progress, print_errors_verbose)
  if columns is None:
    columns = df.columns
  
  return df.apply(lambda series: _translate_series(series, language_code, batch_size, batch_char_limit, wait_time, display_progress, print_errors_verbose) if series.name in columns else series)

def _translate_series(series, language_code='en', batch_size=50, batch_char_limit=10000, wait_time=0, display_progress=False, print_errors_verbose=False):
  """
  <SEMI-PRIVATE> Helper method to translate the entries of a Series to the specified language. Returns the translated Series.
  Specifically, this function just handles null values, passes the data to the next helper function (which does the actual batching
  and translation), and optionally prints errors. 
  As with the translate method: if errors occur during translation, the indices corresponding to those errors will be printed 
  at the end. For those errors, the values in the DataFrame, Series, or list will be the error object from the Azure translation 
  API. These are of type dict, and so that can be used to filter successful translations from unsuccessful ones.
  _translate_series(series, language_code, batch_size, batch_char_limit, wait_time, display_progress, print_errors_verbose)
    
  series: pd.Series
    The series to be translated.
  language_code: str; optional (default='en')
    The language code (not the name!) of the language to translate to.
  batch_size: int; optional (default=50)
    The maximum number of entries which will be sent to the translation API at once.  
  batch_char_limit: int; optional (default = 10,000)
    The maximum number of characters which will be sent to the translation API at once.
  wait_time: int; optional (default=0)
    The amount of time, in seconds, between sending requests to the Azure translation API.
    It is recommended this be non-zero if the dataset is quite large, or entries are long. With an S1 subscription, Azure can only 
    process 11,111 characters per second, so adjusting this wait time is sometimes necessary to ensure the API doesn't generate 
    errors from being overloaded.
    One such error is the throttling error. If this error is encountered, increase this value and the problem should go away.
  display_progress: bool; optional (default=False)
    True if messages should be printed to display the number of entries currently translated.
  print_errors_verbose: bool; optional (default=False)
    True if more detailed error messages should be printed, when encountered.
    
  Returns: pd.Series; The translated data.
  """
    
  # create a copy for the translations and get the entries that need to be translated
  translated_series = series.copy()
  translate_inds = series[series.notna()].index
    
  # translate the entries
  translated_series.loc[translate_inds], error_indices = _batch_and_translate(translated_series.loc[translate_inds], language_code=language_code, batch_size=batch_size, batch_char_limit=batch_char_limit, wait_time=wait_time, display_progress=display_progress, print_errors_verbose=print_errors_verbose)

  # print any errors
  if len(error_indices) > 0:
    display_error_indices(error_indices)
  
  return translated_series

def _batch_and_translate(series_, language_code='en', batch_size=50, batch_char_limit=10000, wait_time=0, display_progress=False, print_errors_verbose=False):
  """
  <SEMI-PRIVATE> Helper method to batch the entries in a Series and then send to the Azure API for translation. 
  Important: At this stage, the data must not contain null values. Returns the translated series and a list of any error indices.
  _batch_and_translate(series, language_code, batch_size, batch_char_limit, wait_time, display_progress, print_errors_verbose)
    
  series: pd.Series
    The series to be translated.
  language_code: str; optional (default='en')
    The language code of the language to translate to.
  batch_size: int; optional (default=50)
    The maximum number of entries which will be sent to the translation API at once.  
  batch_char_limit: int; optional (default = 10,000)
    The maximum number of characters which will be sent to the translation API at once.
  wait_time: int; optional (default=0)
    The amount of time, in seconds, between sending requests to the Azure translation API.
    It is recommended this be non-zero if the dataset is quite large, or entries are long. With an S1 subscription, Azure can only 
    process 11,111 characters per second, so adjusting this wait time is sometimes necessary to ensure the API doesn't generate 
    errors from being overloaded.
    One such error is the throttling error. If this error is encountered, increase this value and the problem should go away.
  display_progress: bool; optional (default=False)
    True if messages should be printed to display the number of entries currently translated.
  print_errors_verbose: bool; optional (default=False)
    True if more detailed error messages should be printed, when encountered.
    
  Returns: (pd.Series, [int]); The translated data and the list of error indices.
  """
  
  batch = Batch(max_size=batch_size, max_chars=batch_char_limit)              # create a Batch object
  error_indices = []
  start_ = next_ = 0
  n = len(series_)
  
  while start_ < n:
    
    next_entry = series_.iloc[next_]
    
    if batch.add(next_entry):              # add entries to the batch until the batch is full
      next_ += 1
      if next_ < n:
        continue
        
    # if the batch contains entries, translate as usual
    # if not, then the current entry is larger than what Azure allows as the maximum number of characters, so the entry must be split before translating
    if not batch.is_empty():
      results, error_response = _translate_batch(batch.get_batch(), language_code)                # translate the batch
      if error_response is not None:                                                             
        for index in series_.index[start_:next_]:                                                 # record any errors
          error_indices.append(index)
        if print_errors_verbose:                                                                  # print errors is desired
          display_batch_error(error_response, list(series_.index[start_:next_]))
      series_.iloc[start_:next_] = results                                                        # add the results 
    else:
      entry_split = pd.Series(sent_tokenize(next_entry))                                          # split into sentences and convert to Series                                     
      entry_translation, errors = _batch_and_translate(entry_split, language_code=language_code, batch_size=batch_size, batch_char_limit=batch_char_limit, wait_time=wait_time)        # translate the series using a recursive call
      if not errors:                                                                                             
        series_.iloc[start_] = (entry_translation + " ").sum()                                                  # add all the translations together with a space in between
      else:                                                                                                     
        error_indices.append(series_.index[start_])
        if print_errors_verbose:
          display_batch_error(entry_translation.iloc[0], [series_.index[start_]])
        series_.iat[start_] = entry_translation.iloc[0]                                                         # add only a single error message for the entry
      next_ += 1
    
    if display_progress:
      display_translation_progress(next_, n)

    start_ = next_
    batch.empty()
    
    sleep(wait_time)                # sleeping is required for large datasets to ensure the Azure API doesn't get overloaded with requests
  
  return series_, error_indices

def _translate_batch(text, language_code):
  """
  <SEMI-PRIVATE> Helper method to format the batch, call the Azure translation API, and handle the response.
  translate_batch(text, language_code)
    
  text: [str]
    The data to be translated.
  language_code: str
    The language code of the language to translate to.
    
  Returns: ([str], [dict]); The translated data and the list of error responses from the Translation API
  If no errors occur, the second tuple argument returned is None.
  """
  azure_translation_url = 'https://api.cognitive.microsofttranslator.com/translate?api-version=3.0'
    
  request_body = _format_batch(text, azure_translation_format)                                    # format the text for the Azure API  
  results = _azure_api_call(request_body, azure_translation_url, additional_params={'to': language_code})           # call the API
  errors = _check_errors(results)                                                                 # check for errors
  error_response = None if not errors else results                                                # create a results message
  results = [result['translations'][0]['text'] for result in results] if not errors else [error_response for entry in text]                             # unpack the results
  return results, error_response
  
def _check_errors(results):
  """
  <SEMI-PRIVATE> Checks whether the results from the Azure API contain errors or not.
  _check_errors(results)
  
  results: [str]
    The results from the Azure translation API call

  Returns: bool; True if the API results contain errors.
  """
  errors = False
  for result in results:
    if 'translations' not in result:
      errors = True
      break
  return errors
  
def _format_batch(text, format_func):
  """
  <SEMI-PRIVATE> Formats each of the text entries in the list into the format
  require by the Azure API. The format function takes an individual 
  text entry, and gives back the necessary format for the API (usually a dict).
  This function was intended to be re-usable and multi-purpose in the sense that 
  if additional formats are required by different API calls, all that's needed 
  is to implement the format_func method on a single entry for that API. 
  This function will then apply that formatting to every entry in a batch.
  _format_batch(text, format_func)
  
  text: [str]
    The data as a list of strings
  format_func: str -> dict
    The function to convert an individual text entry into the format required
    by the desired API.

  Returns: [dict]; The list of entries in the format needed by the API.
  """
  
  return [format_func(entry) for entry in text]

def azure_translation_format(sentence):
  """
  Puts an individual sentence into the required format for the Azure Translation API.
  azure_translation_format(sentence)
  
  sentence: str
    The sentence to be formated
    
  Returns: dict; The dict-format accepted by the Azure Translation API.
  """
  
  return {'Text': sentence}

def _azure_api_call(data, url, headers=api_call_headers, additional_params={}):
  """
  Calls the Azure API with the specifed URL, on the given data with the (optional)
  additional parameters.
  _azure_api_call(data, url, headers, additional_params)
  
  data: [dict]
    The data to be sent to the Azure API
  url: str
    The URL specifying what API service to call
  headers: dict; optional (default={})
    The headers for the API request. This should include the Azure API subscription key
    and the subscription region
  additional_params: {str:str}; optional (default={})    
    A dictionary of additional parameters to pass to the API call.
    
  Returns: JSON object; The response object from the Azure API call.
  """  
  
  if headers == {}:
    display_no_headers_error()
  
  # set the POST request parameters
  params = {
      'api-version': '3.0'
  }
  params.update(additional_params)
  
  # send request
  request = requests.post(url, params=params, headers=headers, json=data)
  response = request.json()
  
  return response


class Batch():
  """Batch class
  This class automates batch maintenance when building batches to send for
  an API call. 
  It allows specification of maximum batch size, and a maximum number of characters
  per batch. Class methods ensure these limits are not violated.
  """
    
  def __init__(self, max_size=MAX_BATCH_SIZE, max_chars=MAX_CHARS):
    self.max_size = max_size
    self.max_chars = max_chars
    
    self.batch = []                                             # initialize an empty batch
    self.num_chars = 0
    
  @property
  def size(self):                                               # get the current batch size
    return len(self.batch)
  
  @property
  def is_full(self):                                            # indicates if the current batch is full or not
    return False if self.size < self.max_size else True
    
  def space_available(self, string):                            # indicates if enough space is available in the batch for 'string'
    return False if self.is_full or self.num_chars + len(string) > self.max_chars else True
  
  def is_empty(self):                                           # is the batch empty
    return self.size == 0
  
  def get_batch(self):                                          # get the current batch
    return self.batch
    
  def add(self, entry):                                         # add an element to the batch
    if not isinstance(entry, str):
      raise ValueError("Only strings can be added to Batch objects")
      return
    elif not self.space_available(entry):
      return False
    
    self.batch.append(entry)
    self.num_chars += len(entry)
    return True
  
  def empty(self):                                              # empty the batch
    self.batch = []
    self.num_chars = 0
   
##############################################################################
### Visualization Methods    
##############################################################################

def display_translation_progress(current, total):
  print("Processed {} of {}".format(current, total))
  
def display_language_error(language):
  print("Error: Language or language code '{}' is not supported by the Azure API.".format(language))
  
def display_batch_violation_error(batch_size, batch_char_limit):
  if batch_size > MAX_BATCH_SIZE:
    print("Error: Requested batch size of {} is too large for the Azure API. (Array size limit: {})".format(batch_size, MAX_BATCH_SIZE))
  else:
    print("Error: Requested batch character limit of {} is too large for the Azure API. (Total character limit: {})".format(batch_char_limit, MAX_CHARS))
    
def display_error_indices(error_indices):
  print("\nError Summary\nNumber of Errors: {}\nAt Indices: {}".format(len(error_indices), error_indices))
  
def display_batch_error(error_response_obj, error_indices):
  print("Translation Error: {} (Code #: {}) for indices {}".format(error_response_obj['error']['message'], error_response_obj['error']['code'], error_indices))
  
def display_no_headers_error():
  print("Error: No headers were specified for the API request.\nRemember to provide your subscription key and region to the 'set_headers' function to set these values for API requests.")
