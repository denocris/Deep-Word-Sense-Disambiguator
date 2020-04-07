import params as pm
import tensorflow as tf
import multiprocessing
import numpy as np
from tensorflow import data
import glob
import csv
import re
import sys



#import spacy
#if pos_tag==True: nlp = spacy.load("it_model0")

# ---------------------------------------------------
# ------------- PIPELINE INPUT FUNCTION -------------
# ---------------------------------------------------

def _list_of_propn():
    propn = []
    remove = ['ha', 'a', 'è', 'e', 'hai', 'ai', 'ho', 'o', 'hanno', 'anno', 'sarà', 'sara', 'sei', '6']
    for name in glob.glob(r'../amb_raw/name_entities/poslayer/*'):
        file = open(name, "r")
        tmp = [ re.sub("'", ' ', line.rstrip().lower()) for line in file ]
        tmp = [ re.sub("-", ' ', line) for line in tmp ]
        tmp = [propn.append(i) for i in tmp if i not in remove]
    propns = r'\b'+r'\b|\b'.join(propn)+r'\b'
    return propns

substitutions=_list_of_propn()




num_words_in_sentence = lambda x: str(x).count(' ') + 1

def parse_tsv_row(tsv_row):
    columns = tf.decode_csv(tsv_row, record_defaults=pm.HEADER_DEFAULTS, field_delim='\t')
    features = {pm.HEADER[0]: columns[0], pm.HEADER[1]: columns[1]}
    #features['length'] = tf.cast(num_words_in_sentence(columns[0]),tf.int64)
    target = features.pop(pm.TARGET_NAME)
    # Uncomment if dataset not already balanced
    #features[WEIGHT_COLUNM_NAME] =  tf.cond( tf.equal(target,'spam'), lambda: 6.6, lambda: 1.0 )
    return features, target


def _parse_and_postag(tsv_row, pos = 'TAG', placeholder = "tannutuva"):

    columns = tf.decode_csv(tsv_row, record_defaults=[['NA'], ['NA']], field_delim='\t')

    postag_col = tf.py_func(lambda x: pos_sentence(x, POS = pos), [columns[0]], tf.string, stateful=False) if pos is not None else columns[0]
    #postag_col = tf.py_func(lambda x: pos_sentence_onlysub(x, POS = pos), [columns[0]], tf.string, stateful=False) if pos is not None else columns[0]

    if placeholder is not None:
        postag_col = tf.regex_replace(postag_col, r"\be\b", placeholder)
        postag_col = tf.regex_replace(postag_col, r"\Bè\B", placeholder)
    else:
        postag_col = postag_col if pos is not None else columns[0]

    features = {'sentence': postag_col, 'class': columns[1]}
    target = features.pop('class')

    return features, target

def _parse_and_postag_v2(tsv_row, pos = 'TAG'):

    columns = tf.decode_csv(tsv_row, record_defaults=[['NA'], ['NA']], field_delim='\t')

    postag_col = tf.py_func(lambda x: pos_sentence(x, POS = pos), [columns[0]], tf.string, stateful=False) if pos is not None else columns[0]

    features = {'sentence': postag_col, 'class': columns[1]}
    target = features.pop('class')

    return features, target

def _placeholder_sub(x, y, placeholder = "tannutuva"):

    features, target = x, y

    features_with_placeholder = tf.regex_replace(features, r"\be\b", placeholder)
    features_with_placeholder = tf.regex_replace(features_with_placeholder, r"\be\b", placeholder)

    features = {'sentence': features_with_placeholder, 'class': target}
    target = features.pop('class')

    return features, target


def parse_label_column(label_string_tensor):
    table = tf.contrib.lookup.index_table_from_tensor(tf.constant(pm.TARGET_LABELS))
    return table.lookup(label_string_tensor)

def input_fn(files_name_pattern, mode=tf.estimator.ModeKeys.EVAL,
                 skip_header_lines=0,
                 num_epochs=1,
                 batch_size=512,
                 pos_tag=False):

    shuffle = True if mode == tf.estimator.ModeKeys.TRAIN else False
    num_threads = multiprocessing.cpu_count() if pm.MULTI_THREADING else 1

    # representing the number of elements from this dataset from which the new dataset will sample.
    buffer_size_prefetch = 2 * batch_size + 1
    buffer_size_shuffle = pm.TRAIN_SIZE #10 * batch_size + 1

    file_names = tf.matching_files(files_name_pattern)
    dataset = data.TextLineDataset(filenames=file_names)
    dataset = dataset.skip(skip_header_lines)

    if shuffle:
        dataset = dataset.shuffle(buffer_size_shuffle)

    #dataset = dataset.map(lambda tsv_row: parse_tsv_row(tsv_row), num_parallel_calls=num_threads)
    if pos_tag:
        dataset = dataset.map(lambda tsv_row: _parse_and_postag(tsv_row, pos = 'TAG', placeholder = "tannutuva"), num_parallel_calls=num_threads)
        #dataset = dataset.map(lambda x, y: _placeholder_sub(x, y, placeholder='tannutuva'), num_parallel_calls=num_threads)
    else:
        dataset = dataset.map(lambda tsv_row: parse_tsv_row(tsv_row), num_parallel_calls=num_threads)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.prefetch(buffer_size_prefetch)
    iterator = dataset.make_one_shot_iterator()
    features, target = iterator.get_next()
    return features, parse_label_column(target)

def process_text(text_feature):

    # Load vocabolary lookup table to map word => word_id
    vocab_table = tf.contrib.lookup.index_table_from_file(vocabulary_file=pm.VOCAB_LIST_FILE,
                                                          num_oov_buckets=1, default_value=-1)
    # Get text feature
    smss = text_feature
    # Split text to words -> this will produce sparse tensor with variable-lengthes (word count) entries
    words = tf.string_split(smss)
    # Convert sparse tensor to dense tensor by padding each entry to match the longest in the batch
    dense_words = tf.sparse_tensor_to_dense(words, default_value=pm.PAD_WORD)
    # Convert word to word_ids via the vocab lookup table
    word_ids = vocab_table.lookup(dense_words)
    # Create a word_ids padding
    padding = tf.constant([[0,0],[0,pm.MAX_DOCUMENT_LENGTH]])
    # Pad all the word_ids entries to the maximum document length
    word_ids_padded = tf.pad(word_ids, padding)
    word_id_vector = tf.slice(word_ids_padded, [0,0], [-1, pm.MAX_DOCUMENT_LENGTH])

    # Return the final word_id_vector
    return word_id_vector


def get_word_index_dict():
    words_index = {}
    with open(pm.VOCAB_LIST_FILE) as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        cnt = 1
        next(tsvfile)
        for row in reader:
            words_index[row[0]] = cnt
            cnt += 1
    return words_index


def load_glove_embeddings(word_index, model):
    embeddings = {}
    for word in model.wv.vocab:
        vectors = model.wv[str(word)]
        #print(vectors)
        embeddings[word] = vectors
    embedding_matrix = np.random.uniform(-1, 1, size=(pm.N_WORDS, 300))
    num_loaded = 0
    for w, i in word_index.items():
        vv = embeddings.get(str(w))
        if vv is not None and i < pm.N_WORDS:
            embedding_matrix[i] = vv
            num_loaded += 1
    #print('Successfully loaded pretrained embeddings for '
          #f'{num_loaded}/{pm.N_WORDS} words.')
    embedding_matrix = embedding_matrix.astype(np.float32)
    return embedding_matrix
