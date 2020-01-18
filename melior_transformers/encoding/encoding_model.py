import logging

from numpy import ndarray

from sentence_transformers import SentenceTransformer
from typing import List


logger = logging.getLogger(__name__)


class SentenceEncoder:

    """ Simple wrapper class around 'sentence-transformers':
     (https://github.com/UKPLab/sentence-transformers/)

    For now only allows to use pre-trained models.

    Eventually fine-tunning should be possible in a similar fashion
    as all the other transformer-tasks
    """

    VALID_MODEL_NAMES = ["bert", "roberta", "distilbert"]
    VALID_MODEL_SIZE = ["base", "large"]
    VALID_DATASET_NAMES = ["nli", "nli-stsb"]
    VALID_POOLING_TYPES = ["max-tokens", "mean-tokens", "cls-token"]

    def __init__(
        self,
        model_name: str = "bert",
        model_size: str = "base",
        dataset_name: str = "nli",
        pooling_type: str = "mean",
        use_cuda: bool = False,
        cuda_device=-1,
    ):
        """
        Initializes a pre-trained Transformer model for Sentence Encoding.

        Args:
            model_name (optional): The type of model (bert, roberta, distilbert)
            model_size (optional): The model size (base, large).
            dataset_name (optional): Dataset of the pre-training (nli, nli-stsb).
            pooling_type (optional): Pooling strategy (max, mean).
            use_cuda (optional): Use GPU if available. Setting to False will
             force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used.
             Will use the first available GPU by default.
        """
        # TODO: Check all parameters are valid before trying to load the model

        self.model_name = (
            f"{model_name}-{model_size}-{dataset_name}-{pooling_type}-tokens"
        )

        try:
            logger.info(f"Loading model '{self.model_name}'")
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            logger.error(f"Error loading model: {e}")

    def encode(
        self, sentences: List[str], batch_size: int = 8, show_progress_bar: bool = False
    ) -> List[ndarray]:
        return self.model.encode(
            sentences, batch_size=batch_size, show_progress_bar=show_progress_bar
        )
