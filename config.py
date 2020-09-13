import transformers

DATA_PATH = "./input/IMDB Dataset.csv"
MAX_TOKEN_LEN = 100
TRAIN_SPLIT = 0.8
BERT_PATH = "./input/bert_base_uncased"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
TRAIN_BATCH_SIZE = 10
NUM_WORKERS = 0
VAL_BATCH_SIZE = 5
SHUFFLE_DATASET = True
RANDOM_SEED = 42
DEVICE = "cuda"
EPOCHS = 20
LEARNING_RATE = 3e-5
MODEL_PATH = 'model.pth'