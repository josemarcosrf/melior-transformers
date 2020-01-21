from sentence_transformers import models

VALID_MODEL_TYPES = ["bert", "roberta", "distilbert"]
VALID_MODEL_SIZE = ["base", "large"]
VALID_DATASET_NAMES = ["nli", "nli-stsb"]
VALID_POOLING_TYPES = ["max", "mean", "cls"]

VALID_MODEL_NAMES = [
    "bert-base-nli-mean-tokens",
    "bert-base-nli-max-tokens",
    "bert-base-nli-cls-token",
    "bert-large-nli-mean-tokens",
    "bert-large-nli-max-tokens",
    "bert-large-nli-cls-token",
    "roberta-base-nli-mean-tokens",
    "roberta-large-nli-mean-tokens",
    "distilbert-base-nli-mean-tokens",
    "bert-base-nli-stsb-mean-tokens",
    "bert-large-nli-stsb-mean-tokens",
    "roberta-base-nli-stsb-mean-tokens",
    "roberta-large-nli-stsb-mean-tokens",
    "distilbert-base-nli-stsb-mean-tokens",
]

MODEL_CLASSES = {
    "bert": models.BERT,
    "xlnet": models.XLNet,
    "roberta": models.RoBERTa,
    "distilbert": models.DistilBERT,
    "albert": models.ALBERT,
    "camembert": models.CamemBERT,
    "t5": models.T5,
    "xlmroberta": models.XLMRoBERTa,
}
