import transformers

DATA_PATH = "./input/IMDB Dataset.csv"
MAX_TOKEN_LEN = 100
SPLIT = 0.2
BERT_PATH = "./input/bert_base_uncased"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)