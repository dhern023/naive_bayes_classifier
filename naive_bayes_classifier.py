# -*- coding: utf-8 -*-
"""
Naive Bayes multi-class classifier inplemented from scratch.
Does not handle Laplacian corrections/smoothing.

Code to accompany "Naive Bayes Classifier" by Luis Serrano
https://www.youtube.com/watch?v=Q8l0Vip5YUw

@author: Diego O. Hernandez
"""

import numpy
import pandas
import pathlib

def calculate_list_product(list_):
    return numpy.array(list_).prod()

def calculate_frequency_average(series):
    try:
        series_averages = series.value_counts() / len(series)
        return series_averages.to_dict()
    except ZeroDivisionError as exception:
        raise exception

def select_rows_wrt_column_value(dataset, column, value):
    mask_contains_value = dataset[column] == value
    return dataset[mask_contains_value]

def check_substring(string, substring, normalize = False):
    """ Normalize compares lower-case """
    if normalize:
        return substring.lower() in string.lower()
    else:
        return substring in string

def construct_series_boolean_via_substring(series_string, metric_substring, substring):
    """ metric_substring should be able to take in two values """
    series_boolean = series_string.apply(metric_substring, args=(substring,))
    return series_boolean

def calculate_likelihoods_categorical_features(dataset, column_label, list_column_categorical):
    """ 
    Likelihood is P( feature | classification ) 
    Calculated directly using the frequency.
    """
    dict_likelihood_per_class = {}
    for classification in dataset[column_label].unique():
        dict_likelihood_per_class.setdefault(classification, [])
        dataset_classified = select_rows_wrt_column_value(dataset, column_label, classification)
        for feature in list_column_categorical:
            dict_probability_conditional = calculate_frequency_average(dataset_classified[feature])
            dict_likelihood_per_class[classification].append(dict_probability_conditional)

    return dict_likelihood_per_class

def calculate_posterior_given_class(list_likelihoods, prior):
    likelihood = calculate_list_product(list_likelihoods)
    return likelihood * prior

def calculate_unmarginalized_posteriors(list_features, dict_likelihoods, dict_priors):
    """ 
    P( Class | Features ) = Product P(Feature | Event) 
    We call these "unmarginalized" because 
    we don't divide by the marginal yet
    """
    dict_posteriors_unmarginalized = {}
    for key_class, prior in dict_priors.items():
        list_likelihoods = []
        for index, feature in enumerate(list_features):
            likelihood = dict_likelihoods[key_class][index][feature]
            list_likelihoods.append(likelihood)
        dict_posteriors_unmarginalized[key_class] = calculate_posterior_given_class(list_likelihoods, prior)

    return dict_posteriors_unmarginalized

def calculate_marginalized_posteriors(dict_unmarginalized_posteriors):
    """ Marginal = Sum( unmarginalized posteriors ) """
    dict_marginalized_posteriors = {}
    
    probability_marginal = sum(dict_unmarginalized_posteriors.values())
    for classification, posterior in dict_unmarginalized_posteriors.items():
         dict_marginalized_posteriors[classification] = posterior / probability_marginal
    
    return dict_marginalized_posteriors

def get_prediction(dict_marginalized_posteriors):
    return max(dict_marginalized_posteriors, key=dict_marginalized_posteriors.get)

def process_spam_classifier_dataframe(dataset_spam, column_email, list_strings_spammy):
    """
    Constructs several categorical boolean columns based 
    on if email body contains substring from list.
    """
    
    dataframe_spam_processed = dataset_spam.copy()
    
    # Creates categorical columns if email body contains spammy words
    for string in list_strings_spammy:
        column_name = "contains_{}".format(string)
        dataframe_spam_processed[column_name] = construct_series_boolean_via_substring(dataframe_spam_processed[column_email], check_substring, string)

    return dataframe_spam_processed
    
if __name__ == "__main__":
    # Environment variables
    dir_data = pathlib.Path.cwd() / 'data'
    name_dataset = "test_spam.csv"
    dataset = pandas.read_csv(dir_data / name_dataset)

    # Spam classifier parameters
    column_label = "classification"
    column_feature_discrete = "email_body"
    list_spammy_words = ["buy", "cheap"]

    columns_feature_categorical = ["contains_{}".format(string) for string in list_spammy_words]
    dataframe_spam = process_spam_classifier_dataframe(dataset, column_feature_discrete, list_spammy_words)

    dict_probability_prior = calculate_frequency_average(dataframe_spam[column_label])
    dict_probability_likelihood = calculate_likelihoods_categorical_features(dataframe_spam, column_label, columns_feature_categorical)
    
    # Contains buy, contains cheap
    features = [True, True]

    dict_posteriors_unmarginalized = calculate_unmarginalized_posteriors(
        features, dict_probability_likelihood, dict_probability_prior)

    dict_classification_probabilities = calculate_marginalized_posteriors(dict_posteriors_unmarginalized)
    prediction = get_prediction(dict_classification_probabilities)
    string_output_classification = "Classify email as spam?: {}".format(prediction)
    string_output_probability = "Probability email is spam given features?: {}".format(dict_classification_probabilities[prediction])

    print(string_output_classification)
    print(string_output_probability)