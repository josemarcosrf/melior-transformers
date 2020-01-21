import logging
from typing import List

from numpy import ndarray

from melior_transformers.encoding.constants import (
    VALID_DATASET_NAMES,
    VALID_MODEL_NAMES,
    VALID_MODEL_SIZE,
    VALID_MODEL_TYPES,
    VALID_POOLING_TYPES,
)
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class SiameseSentenceEncoder:

    """ Simple wrapper class around 'sentence-transformers':
     (https://github.com/UKPLab/sentence-transformers/)

    For now only allows to use pre-trained models.

    Eventually fine-tunning should be possible in a similar fashion
    as all the other transformer-tasks
    """

    def __init__(
        self,
        model_type: str = "bert",
        model_size: str = "base",
        dataset_name: str = "nli",
        pooling_type: str = "mean",
        use_cuda: bool = False,
        cuda_device=-1,
    ):
        """
        Initializes a pre-trained Transformer model for Sentence Encoding.

        Args:
            model_type (optional): The type of model (bert, roberta, distilbert)
            model_size (optional): The model size (base, large).
            dataset_name (optional): Dataset of the pre-training (nli, nli-stsb).
            pooling_type (optional): Pooling strategy (max, mean, cls).
            use_cuda (optional): Use GPU if available. Setting to False will
             force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used.
             Will use the first available GPU by default.
        Returns:
            None
        """

        if model_type not in VALID_MODEL_TYPES:
            raise ValueError(f"Model type {model_type} doesn't exist.")
        if model_size not in VALID_MODEL_SIZE:
            raise ValueError(f"Model size {model_size} doesn't exist.")
        if dataset_name not in VALID_DATASET_NAMES:
            raise ValueError(f"Dataset name {dataset_name} doesn't exist.")
        if pooling_type not in VALID_POOLING_TYPES:
            raise ValueError(f"Pooling type {pooling_type} doesn't exist.")

        self.model_name = (
            f"{model_type}-{model_size}-{dataset_name}-{pooling_type}-tokens"
        )

        if self.model_name not in VALID_MODEL_NAMES:
            raise ValueError(
                f"Model name {self.model_name} doesn't exist. "
                f"\n Please select one of the aviables {VALID_MODEL_NAMES}"
            )

        try:
            logger.info(f"Loading model '{self.model_name}'")
            self.model = SentenceTransformer(
                self.model_name, use_cuda=use_cuda, cuda_device=cuda_device
            )
        except Exception as e:
            raise ValueError(f"Error loading model: {e}")

    def encode(
        self, sentences: List[str], batch_size: int = 8, show_progress_bar: bool = False
    ) -> List[ndarray]:
        """
        Extract sentence embeddings from the selected model.

        Args:
            sentences: List of sentences to extract embeddings.
            batch_size (optional): Batch size used for the computation
            show_progress_bar (optional): Output a progress bar when encode sentences
        Returns:
           List with ndarrays of the embeddings for each sentence.
        """

        return self.model.encode(
            sentences, batch_size=batch_size, show_progress_bar=show_progress_bar
        )
