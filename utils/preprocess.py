import gensim
import gensim.corpora as corpora
import nltk


def replace_null(col, replace_text):
    col = col.copy()
    return col.fillna(replace_text)


def remove_stop_words(col, stop_words):
    pat = r'\b(?:{})\b'.format('|'.join(stop_words))
    col = col.str.replace(pat, '')
    col = col.str.replace(r'\s+', ' ')
    return col


def remove_short_words(col, length):
    col = col.copy()
    t = col.str.split(expand=True).stack()
    return t.loc[t.str.len() >= length].groupby(level=0).apply(' '.join)


def remove_digits(col):
    col = col.copy()
    return col.str.replace('\d+', '')


def filter_reviews_rating(df, threshold):
    """
    Filters reviews with rating below a threshold (normalized)
    :param df: dataframe of reviews
    :param threshold: float from 0 to 1
    :return: filtered reviews dataframe
    """
    df = df.copy()
    return df[df['cleaned_ratings'] <= threshold]


def filter_short_words(df, threshold):
    """
    Filters reviews with word count above threshold
    :param df: dataframe of reviews
    :param threshold: int
    :return: filtered reviews dataframe
    """
    return df[df['cleaned_text_word_count'] > threshold]


def filter_pos_tags(text, tag_lst):
    """
    Filters reviews by pos tags (upenn tree bank)
    :param text: string to filter
    :param tag_lst: list of tags with first character to filter i.e. ['N', 'V'] filters all nouns and verbs.
    :return: filtered sentence (string)
    """
    tokens = nltk.word_tokenize(text)
    tags = nltk.pos_tag(tokens)
    filtered_sent = [word for word, tag in tags if tag[0] in tag_lst]
    return ' '.join(filtered_sent)


def sent_to_words(sentences):
    for sentence in sentences:
        yield gensim.utils.simple_preprocess(str(sentence))


def generate_bigrams(words):
    bigram = gensim.models.Phrases(words, min_count=5, threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return [bigram_mod[doc] for doc in words]


def generate_dictionary(data):
    return corpora.Dictionary(data)