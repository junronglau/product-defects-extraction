import argparse
from utils.preprocess import replace_null
from utils.preprocess import remove_short_words
from utils.preprocess import remove_digits
from utils.preprocess import filter_pos_tags
from utils.preprocess import filter_reviews_rating
from utils.preprocess import remove_stop_words

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument(
        '-c', '--config',
        dest='config',
        metavar='C',
        default='configs/config.json',
        help='The Configuration file')

    args = argparser.parse_args()
    return args


def app_preprocess(df):
    def generate_stop_word_list():
        with open("data/uc3/labels_generator/stop_words.txt") as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        return content

    df['cleaned_text'] = remove_digits(df['cleaned_text'])
    df['cleaned_text'] = remove_short_words(df['cleaned_text'], 3)
    df['cleaned_text'] = df['cleaned_text'].astype(str).apply(filter_pos_tags, tag_lst=['N', 'V', 'J'])
    df['cleaned_text'] = remove_stop_words(df['cleaned_text'], generate_stop_word_list())
    df['cleaned_text'] = replace_null(df['cleaned_text'], '')

    return df