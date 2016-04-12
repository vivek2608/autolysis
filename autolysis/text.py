import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def features_sparse_matrix(series, top_words=50, count_repeat=True, **kwargs):
    '''
    Returns feature matrix of the

    Parameters
    ------------
    series: pd.Series,
        Pandas series which contains the text
    top_words: int,
        Pick only top `n` words based on frequency
    count_type: boolean,
        If False, just ignore the count within a row. If the word appears
        more than once, consider only once while counting. Default is True.

    '''
    stop_words = 'english' if 'stop_words' not in kwargs else kwargs['stop_words']
    token_pattern = '[a-zA-Z]{3,}' if 'token_pattern' not in kwargs else kwargs['token_pattern']
    kwargs.pop('stop_words', None)
    kwargs.pop('token_pattern', None)
    count_vec = CountVectorizer(stop_words=stop_words, token_pattern=token_pattern, **kwargs)
    sparse_matrix = count_vec.fit_transform(series.dropna())
    features_freq = np.array(sparse_matrix.sum(axis=0))[0]
    features_top_index = features_freq.argsort()[-top_words:][::-1]
    sparse_matrix_subset = sparse_matrix[:, features_top_index].toarray()
    features_name = np.array(count_vec.get_feature_names())[features_top_index]
    data_features = pd.DataFrame(sparse_matrix_subset, columns=features_name)
    if not count_repeat:
        data_features[data_features > 0] = 1
    return data_features
