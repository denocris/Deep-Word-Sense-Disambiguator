import os, sys
import models
import tensorflow as tf


# ------------------------------------
# ------------ ANN MODEL -------------
# ------------------------------------
##
#MODEL_FN = models.lstm_model_fn
MODEL_FN = models.bidir_lstm_model_fn
#MODEL_FN = models.cnn_model_fn
#MODEL_FN = models.cnn_lstm_model_fn


PRINT_SHAPE = False
# ------------------------------------
# ----- SETUP TRAINING----------------
# ------------------------------------
RESUME_TRAINING = False
MULTI_THREADING = True
TRAIN_SIZE = int(str(sys.argv[7]))
NUM_EPOCHS = 70
BATCH_SIZE = 256
EVAL_AFTER_SEC = 60
TOTAL_STEPS = int((TRAIN_SIZE/BATCH_SIZE)*NUM_EPOCHS)

# ------------------------------------
# ------------- METADATA -------------
# ------------------------------------
PAD_WORD = '#=KS=#'
HEADER = ['sentence', 'class']
HEADER_DEFAULTS = [['NA'], ['NA']]
TEXT_FEATURE_NAME = 'sentence'
TARGET_NAME = 'class'
TARGET_LABELS = ['0', '1']
#WEIGHT_COLUNM_NAME = 'weight'

# ------------------------------------
# ----------- GLOVE EMBEDDING --------
# ------------------------------------
GLOVE_ACTIVE=True
TRAINABLE_EMB=True
MAX_DOCUMENT_LENGTH = 20
EMBEDDING_SIZE = 300 if GLOVE_ACTIVE else 8

# ------------------------------------
# ------------- TRAINING PARAMS ------------
# ------------------------------------
LEARNING_RATE = 0.00005 #float(str(sys.argv[8])) #0.0005 #0.0001
DECAY_LEARNING_RATE_ACTIVE = True
# For LSTM0
FORGET_BIAS = 1.0
# For LSTM0
DROPOUT_RATE = 0.5 #0.5
# For LSTM it refers to the size of the Cell, for CNN model instead are the FC layers
HIDDEN_UNITS = [32] #[96, 64, 16], None
# For CNN, kernel size
WINDOW_SIZE = 3
# For CNN, number of filters (i.e. feature maps)
FILTERS = 64

# ------------------------------------
# ------------- MODEL DIR ------------
# ------------------------------------
MODEL_NAME = str(sys.argv[1])
#MODEL_DIR = os.path.join(os.getcwd(),'/home/asr/rd_ssDataLearning/trained_ssDataLearning/{}'.format(MODEL_NAME))
#MODEL_DIR = os.path.join(os.getcwd(),str(sys.argv[2])+'{}'.format(MODEL_NAME))
MODEL_DIR = str(sys.argv[2])+'{}'.format(MODEL_NAME)

TRAIN_DATA_PATH = str(sys.argv[3])
VALID_DATA_PATH = str(sys.argv[4])
VOCAB_PATH = str(sys.argv[5])
NUM_WORDS_PATH = str(sys.argv[6])
# ------------------------------------
# ------- TRAIN & VALID PATH ---------
# ------------------------------------


TRAIN_DATA_FILES_PATTERN = TRAIN_DATA_PATH
VALID_DATA_FILES_PATTERN = VALID_DATA_PATH
VOCAB_LIST_FILE = VOCAB_PATH
N_WORDS_FILE = NUM_WORDS_PATH



with open(N_WORDS_FILE) as file:
    N_WORDS = int(file.read())+2
