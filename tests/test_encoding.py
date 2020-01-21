import pytest
import scipy

from melior_transformers.encoding.encoding_model import SentenceEncoder
from melior_transformers.encoding.siamese_encoding_model import SiameseSentenceEncoder

corpus = [
    "A man is eating food.",
    "A man is eating a piece of bread.",
    "The girl is carrying a baby.",
    "A man is riding a horse.",
    "A woman is playing violin.",
    "Two men pushed carts through the woods.",
    "A man is riding a white horse on an enclosed ground.",
    "A monkey is playing drums.",
    "A cheetah is running behind its prey.",
]

queries = [
    "A man is eating pasta.",
    "Someone in a gorilla costume is playing a set of drums.",
    "A cheetah chases prey on across a field.",
]


@pytest.mark.parametrize(
    "model_type, model_name",
    [
        ("bert", "bert-base-uncased"),
        ("xlnet", "xlnet-base-cased"),
        ("xlm", "xlm-mlm-17-1280"),
        ("roberta", "roberta-base"),
        ("distilbert", "distilbert-base-uncased"),
        ("albert", "albert-base-v1"),
        ("camembert", "camembert-base"),
        ("xlmroberta", "xlm-roberta-base"),
    ],
)
def test_encoding_model(model_type, model_name):
    se = SentenceEncoder(
        model_type, model_name, args={"max_seq_length": 128}, random_seed=1
    )
    se.encode(corpus)


@pytest.mark.parametrize(
    "model_type,model_size,dataset_name,pooling_type",
    [
        ("bert", "base", "nli", "max"),
        ("roberta", "base", "nli", "mean"),
        ("distilbert", "base", "nli", "mean"),
        ("bert", "base", "nli-stsb", "mean"),
        ("distilbert", "base", "nli-stsb", "mean"),
    ],
)
def test_siamese_encoding_model(model_type, model_size, dataset_name, pooling_type):
    se = SiameseSentenceEncoder(
        model_type=model_type,
        model_size=model_size,
        dataset_name=dataset_name,
        pooling_type=pooling_type,
        use_cuda=False,
        random_seed=1,
    )
    corpus_embeddings = se.encode(corpus)
    query_embeddings = se.encode(queries)

    for query, query_embedding in zip(queries, query_embeddings):
        distances = scipy.spatial.distance.cdist(
            [query_embedding], corpus_embeddings, "cosine"
        )[0]
        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])
        if query == "A man is eating pasta":
            assert corpus[results[0][0]] == corpus[1]
        if query == "Someone in a gorilla costume is playing a set of drums.":
            assert corpus[results[0][0]] == corpus[7]
        if query == "A cheetah chases prey on across a field.":
            assert corpus[results[0][0]] == corpus[8]
        print(results)
