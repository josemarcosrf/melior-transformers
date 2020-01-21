import logging
from typing import Dict, List

import numpy as np
import torch
from numpy import ndarray
from sentence_transformers import SentenceTransformer, models

from melior_transformers.config.global_args import global_args
from melior_transformers.encoding.constants import MODEL_CLASSES

logger = logging.getLogger(__name__)


class SentenceEncoder:
    """ Simple wrapper class around 'sentence-transformers':
     (https://github.com/UKPLab/sentence-transformers/) that
     allow us to easly-extract embeddings from pre-trained models.

     You can find the full list of models here:
     https://huggingface.co/transformers/pretrained_models.html
    """

    def __init__(
        self,
        model_type: str = "bert",
        model_name: str = "bert-base-uncased",
        args: Dict = None,
        use_cuda: bool = False,
        random_seed: int = None,
    ):

        """
        Initializes a pre-trained Transformer model for Sentence Encoding.

        Args:
            model_type (optional): The type of model.
            model_size (optional): The model name.
            args (optional): Aditional arguments to configure embeddigs extraction.
            use_cuda (optional): Use GPU if available. Setting to False will
             force model to use CPU only.
        Returns:
            None
        """

        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

        if use_cuda:
            device = "cuda"
        else:
            device = "cpu"

        self.args = {
            # Model config
            "max_seq_length": 128,
            "do_lower_case": False,
            # Model config
            "pooling_mode_mean_tokens": True,
            "pooling_mode_cls_token": False,
            "pooling_mode_max_tokens": False,
            "pooling_mode_mean_sqrt_len_tokens": False,
        }

        if args is not None:
            self.args.update(args)

        if model_type not in MODEL_CLASSES:
            raise ValueError(
                f"Model type {model_type} doesn't exist."
                f"\nPlease select one of the follwing: {MODEL_CLASSES.keys()}"
            )

        try:
            logger.info(f"Loading model '{model_name}'")

            word_embedding_model = MODEL_CLASSES[model_type](
                model_name_or_path=model_name,
                max_seq_length=self.args["max_seq_length"],
                do_lower_case=self.args["do_lower_case"],
            )

            pooling_model = models.Pooling(
                word_embedding_model.get_word_embedding_dimension(),
                pooling_mode_mean_tokens=self.args["pooling_mode_mean_tokens"],
                pooling_mode_cls_token=self.args["pooling_mode_cls_token"],
                pooling_mode_max_tokens=self.args["pooling_mode_max_tokens"],
                pooling_mode_mean_sqrt_len_tokens=self.args[
                    "pooling_mode_mean_sqrt_len_tokens"
                ],
            )

            self.encoder_model = SentenceTransformer(
                modules=[word_embedding_model, pooling_model], device=device
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

        return self.encoder_model.encode(
            sentences, batch_size=batch_size, show_progress_bar=show_progress_bar
        )
