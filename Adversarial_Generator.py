import os, time, torch
import sys

from transformers import logging as hfl
import pandas as pd
from transformers import pipeline
import logging, warnings, tensorflow as tf
from huggingface_hub.utils import disable_progress_bars
from tqdm import tqdm
from nltk import word_tokenize
import re, nltk, ssl
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt_tab')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
disable_progress_bars()
start_time = time.time()
hfl.set_verbosity_error()
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel('INFO')
tf.get_logger().setLevel(logging.ERROR)
os.environ['TOKENIZERS_PARALLELISM'] = 'False'
warnings.simplefilter(action='ignore', category=Warning)
warnings.simplefilter(action='ignore', category=FutureWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
sa = pipeline('text-classification', model='CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment', return_all_scores=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

stop = set(nltk.corpus.stopwords.words("arabic"))

substitution_groups = {
    'ت': ['ث', 'ب', 'ن'],
    'ب': ['ت', 'ث', 'ن'],
    'ث': ['ت', 'ب', 'ن'],
    'ن': ['ب', 'ت', 'ث', 'ي'],
    'ي': ['ى'],
    'ى': ['ي'],
    'ف': ['ق'],
    'ق': ['ف'],
    'ر': ['ز'],
    'ز': ['ر'],
    'ع': ['غ'],
    'غ': ['ع'],
    'ض': ['ص'],
    'ه': ['ة'],
    'ص': ['ض'],
    'ة': ['ه'],
}


def tokenize(text):
    tokenized_text = []
    from nltk.tokenize import WhitespaceTokenizer
    tokenized_tokens = WhitespaceTokenizer().tokenize(text)
    for token in tokenized_tokens:
        if token not in tokenized_text:
            tokenized_text.append(token)
    return tokenized_text


def clean(text):
    arabicPunctuations = [".", "`", "؛", "<", ">", "(", ")", "*", "&", "^", "%", "]", "[", ",", "–",
                          "ـ", "،", "/", ":", "؟", ".", "'", "{", "}", "~", "|", "!", "”", "…", "“"]

    def remove_punctuation(text):
        cleanText = ''
        for i in text:
            if i not in arabicPunctuations:
                cleanText = cleanText + '' + i
        return cleanText

    def remove_emoji(text):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"
                                   u"\U0001F300-\U0001F5FF"
                                   u"\U0001F680-\U0001F6FF"
                                   u"\U0001F1E0-\U0001F1FF"
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
        return text

    def remove_stopwords(text):
        temptext = word_tokenize(text)
        text = " ".join([w for w in temptext if not w in stop and len(w) >= 2])
        return text

    def remove_noise(text):
        text = re.sub('\s+', ' ', text)
        text = re.sub('[^\u0621-\u064A\u0660-\u0669 ]+', '',
                      text)
        return text.strip()

    text = remove_noise(text)
    text = remove_emoji(text)
    text = remove_stopwords(text)
    text = remove_punctuation(text)

    return text


def tag(text):
    from transformers import pipeline
    tagger = pipeline('token-classification', model='bert-base-arabic-camelbert-ca-pos-egy', device=device)

    results = tagger(text)
    return results


def check(text1, text2):
    """ check similarity between two text """

    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', device="gpu")
    encoded_text1 = model.encode(text1)
    encoded_text2 = model.encode(text2)

    similarity = cosine_similarity([encoded_text1], [encoded_text2])[0][0]

    return similarity


def generate_adversarial_dot_based(word, substitution_groups):
    """
    Generates adversarial examples by substituting one character in the word with a visually similar character,
    focusing on characters differentiated by dots.

    Args:
    - word (str): The original Arabic word.
    - substitution_groups (dict): A dictionary where keys are characters and values are lists of characters that
      are visually similar except for the dots.

    Returns:
    - list: A list of adversarial words, each with exactly one character changed.
    """
    adversarial_words = []

    # Iterate over each character in the word
    for i, char in enumerate(word):
        # Check if the character has similar characters based on the dot difference
        if char in substitution_groups:
            # Substitute with each similar character
            for similar_char in substitution_groups[char]:
                # Create a new word with one substituted character
                new_word = word[:i] + similar_char + word[i + 1:]
                adversarial_words.append(new_word)

    return adversarial_words


def apply_attack_1(idx_miw, sentences_list, adversarial_examples):
    """"""
    base_pos, base_neg, base_neu = compute_sent(" ".join(sentences_list))
    print(f" \n the original sentence {' '.join(sentences_list)}\n")

    sent_list = []
    # i need to save the indx and if the diff is larger than the stored one ad store the tmp sent. if that condition
    # is true
    max_diff = 0
    max_df_pos = 0
    max_df_neg = 0
    max_df_neu = 0
    for adv in range(0, len(adversarial_examples)):
        replaced_word = sentences_list[idx_miw]
        sentences_list[idx_miw] = adversarial_examples[adv]
        tmp_pos, tmp_neg, tmp_neu = compute_sent(" ".join(sentences_list))
        sentences_list[idx_miw] = replaced_word

        differences = compute_differences(base_pos, base_neg, base_neu, tmp_pos, tmp_neg, tmp_neu)

        sent_list.append([adv, differences])

        if differences > max_diff:
            max_df_pos, max_df_neg, max_df_neu = tmp_pos, tmp_neg, tmp_neu

    idx_most_effective_attack = [max(sent_list, key=lambda item: item[1])[0]][0]

    most_effective_attack = adversarial_examples[idx_most_effective_attack]

    sentences_list[idx_miw] = most_effective_attack

    return most_effective_attack, ' '.join(sentences_list), sent_list[idx_most_effective_attack][
        1], base_pos, base_neg, base_neu, max_df_pos, max_df_neg, max_df_neu


def apply_attack_2(idx_miw, sentences_list, adversarial_examples, base_pos, base_neg, base_neu):
    """"""

    sent_list = []
    # i need to save the indx and if the diff is larger than the stored one ad store the tmp sent. if that condition
    # is true
    max_diff = 0
    max_df_pos = 0
    max_df_neg = 0
    max_df_neu = 0
    for adv in range(0, len(adversarial_examples)):
        replaced_word = sentences_list[idx_miw]
        sentences_list[idx_miw] = adversarial_examples[adv]
        tmp_pos, tmp_neg, tmp_neu = compute_sent(" ".join(sentences_list))
        sentences_list[idx_miw] = replaced_word
        differences = compute_differences(base_pos, base_neg, base_neu, tmp_pos, tmp_neg, tmp_neu)
        sent_list.append([adv, differences])
        if differences > max_diff:
            max_df_pos, max_df_neg, max_df_neu = tmp_pos, tmp_neg, tmp_neu

    idx_most_effective_attack = [max(sent_list, key=lambda item: item[1])[0]][0]

    most_effective_attack = adversarial_examples[idx_most_effective_attack]

    sentences_list[idx_miw] = most_effective_attack

    return most_effective_attack, ' '.join(sentences_list), sent_list[idx_most_effective_attack][
        1], max_df_pos, max_df_neg, max_df_neu


def compute_differences(base_pos, base_neg, base_neu, tmp_pos, tmp_neg, tmp_neu):
    # Calculate the absolute differences
    diff_pos = abs(base_pos - tmp_pos)
    diff_neg = abs(base_neg - tmp_neg)
    diff_neu = abs(base_neu - tmp_neu)

    return diff_pos + diff_neg + diff_neu


def compute_sent(sentences):
    """compute the sent for the passed text """
    pred = sa.predict(sentences)
    pos = pred[0][0]
    neg = pred[0][1]
    nut = pred[0][2]
    txt_pos = pos["score"]
    txt_neg = neg["score"]
    txt_nut = nut["score"]
    return txt_pos, txt_neg, txt_nut


def find_most_important_vulnerable_word(sentences):
    """by find the difference between the sent. for the base sentence and the sentence without one word to find the
    MIW """

    base_pos, base_neg, base_neu = compute_sent(sentences)
    sentences_list = sentences.split(" ")
    list_of_sent = []
    for i in range(len(sentences_list) - 1, 0, -1):
        re_insert = sentences_list[i]
        del sentences_list[i]
        tmp_pos, tmp_neg, tmp_neu = compute_sent(' '.join(sentences_list))
        list_of_sent.append([i, compute_differences(base_pos, base_neg, base_neu, tmp_pos, tmp_neg, tmp_neu)])
        sentences_list.insert(i, re_insert)

    idx_miw = [max(list_of_sent, key=lambda item: item[1])[0]][0]
    miw = sentences.split(" ")[max(list_of_sent, key=lambda item: item[1])[0]]

    # check if the miw has any of the substitution_groups.keys to make sure it's vulnerable
    while not (any(key in miw for key in substitution_groups.keys())):
        del list_of_sent[idx_miw]
        idx_miw = [max(list_of_sent, key=lambda item: item[1])[0]][0]
        miw = sentences.split(" ")[max(list_of_sent, key=lambda item: item[1])[0]]

    print(f"\n {idx_miw} is the index of the most important word: {miw}\n")
    return idx_miw, miw, sentences_list
def run_DOT_Attack(inp):
    cln_txt = clean(inp)
    try:
        temp_cln_txt = cln_txt.split(" ")
        if len(temp_cln_txt) > 10:
            temp_cln_txt = temp_cln_txt[:]
            cln_txt = " ".join(temp_cln_txt)

        if 5 <= len(cln_txt.split(" ")):
            idx_first_miw, first_miw, first_sentences_list = find_most_important_vulnerable_word(cln_txt)

            # Find second MIW
            first_sentences_list.pop(idx_first_miw)

            txt_wo_first_word = " ".join(first_sentences_list)

            first_sentences_list.insert(idx_first_miw, first_miw)

            _, second_miw, second_sentences_list = find_most_important_vulnerable_word(txt_wo_first_word)

            idx_second_miw = first_sentences_list.index(second_miw)

            # Find third MIW

            first_sentences_list.pop(idx_first_miw)

            first_sentences_list.pop(idx_second_miw)

            txt_wo_first_second_words = " ".join(first_sentences_list)

            first_sentences_list.insert(idx_first_miw - 1, first_miw)

            first_sentences_list.insert(idx_second_miw, second_miw)

            _, third_miw, third_sentences_list = find_most_important_vulnerable_word(
                txt_wo_first_second_words)

            idx_third_miw = first_sentences_list.index(third_miw)

            print("first sentence:", first_sentences_list, "\n", first_miw, "\t", idx_first_miw)
            print("second sentence:", second_sentences_list, "\n", second_miw, "\t", idx_second_miw)
            print("third sentence:", third_sentences_list, "\n", third_miw, "\t", idx_third_miw)

            first_adversarial_examples = generate_adversarial_dot_based(first_miw, substitution_groups)
            second_adversarial_examples = generate_adversarial_dot_based(second_miw, substitution_groups)
            third_adversarial_examples = generate_adversarial_dot_based(third_miw, substitution_groups)

            print(f"\n the first adversarial examples: \t {first_adversarial_examples}")
            print(f"\n the second adversarial examples: \t {second_adversarial_examples}")
            print(f"\n the third adversarial examples: \t {third_adversarial_examples}")

            mea1, adv_stm1, _, base_pos, base_neg, base_neu, max_df_pos1, max_df_neg1, max_df_neu1 = apply_attack_1(
                idx_first_miw,
                first_sentences_list,
                first_adversarial_examples)

            mea2, adv_stm2, _, max_df_pos2, max_df_neg2, max_df_neu2 = apply_attack_2(
                idx_second_miw,
                adv_stm1.split(" "),
                second_adversarial_examples, base_pos, base_neg, base_neu)

            mea3, adv_stm3, _, max_df_pos3, max_df_neg3, max_df_neu3 = apply_attack_2(
                idx_third_miw,
                adv_stm2.split(" "),
                third_adversarial_examples, base_pos, base_neg, base_neu)

            most_effective_words = mea1 + "," + mea2 + "," + mea3

            print(f"Adv sentence 1: \t {adv_stm1} \n Adv sentence 2 \t {adv_stm2} \n Adv sentence 3 \t {adv_stm3}")
            print("most_effective_words: \t", most_effective_words)
    except:
        print("Error occured")


run_DOT_Attack("بذلت مجهودا جبارا في محاولة قراءة الكتاب لكنني فشلت القصة غريبة جداً لدرجة أنني توقفت ولم أستطع إكمال القراءة")
