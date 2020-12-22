import gensim
import gensim.corpora as corpora


def filter_reviews_rating(df, threshold):
    """
    Filters reviews with rating below a threshold (normalized)
    :param df: dataframe of reviews
    :param threshold: float from 0 to 1
    :return: filtered reviews dataframe
    """
    return df[df['ratings'] <= threshold]


def sent_to_words(sentences):
    for sentence in sentences:
        yield gensim.utils.simple_preprocess(str(sentence))


def generate_bigrams(words):
    bigram = gensim.models.Phrases(words, min_count=5, threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return [bigram_mod[doc] for doc in words]


def generate_dictionary(data):
    return corpora.Dictionary(data)