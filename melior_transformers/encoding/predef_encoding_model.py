import logging

from numpy import ndarray

from sentence_transformers import models, SentenceTransformer
from melior_transformers.config.global_args import global_args
from typing import List


logger = logging.getLogger(__name__)


class SentenceEncoder:

    VALID_MODEL_NAMES = ["bert", "roberta", "distilbert"]
    VALID_MODEL_SIZE = ["base", "large"]
    VALID_DATASET_NAMES = ["nli", "nli-stsb"]
    VALID_POOLING_TYPES = ["max-tokens", "mean-tokens", "cls-token"]

    def __init__(
        self,
        model_type: str = "bert",
        model_name: str = "base",
        args: Dict = None,
        use_cuda: bool = False,
        cuda_device=-1,
    ):
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

        self.args = {
            "max_seq_length": 128,
            "do_lower_case": False,
        }

        if args is not None:
            self.args.update(args)

        print(self.args)

        word_embedding_model = MODEL_CLASSES[model_type](
            model_name_or_path=model_name,
            max_seq_length=self.args["max_seq_length"],
            do_lower_case=self.args["do_lower_case"],
        )

        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False,
        )

        self.encoder_model = SentenceTransformer(
            modules=[word_embedding_model, pooling_model],
        )

    def encode(
        self, sentences: List[str], batch_size: int = 8, show_progress_bar: bool = False
    ) -> List[ndarray]:

        return self.encoder_model.encode(
            sentences, batch_size=batch_size, show_progress_bar=show_progress_bar
        )


if __name__ == "__main__":
    se = SentenceEncoder("bert", "bert-base-uncased", args={"max_seq_length": 1024})
    sentences = se.encode(["How are you?", "Are you ok?"],)
    print(sentences[0].size)
    print(sentences[1].size)
