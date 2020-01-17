from melior_transformers.classification import ClassificationModel
import pandas as pd


# Train and Evaluation data needs to be in a Pandas Dataframe of two columns. The first column is the text with type str, and the second column is the label with type int.
train_data = [
    ["Example sentence belonging to class 1", 1],
    ["Example sentence belonging to class 0", 0],
]
train_df = pd.DataFrame(train_data)

eval_data = [
    ["Example eval sentence belonging to class 1", 1],
    ["Example eval sentence belonging to class 0", 0],
]
eval_df = pd.DataFrame(eval_data)

# Create a ClassificationModel
model = ClassificationModel(
    "roberta",
    "roberta-base",
    args={"reprocess_input_data": True, "overwrite_output_dir": True},
    use_cuda=False,
)

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)
print(result)
print(model_outputs)
