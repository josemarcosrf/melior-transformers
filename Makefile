install:
	pip install -e .
	pip install -r requirements-dev.txt
	pip list

clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f  {} +

formatter:
	black melior_transformers tests \
		--exclude melior_transformers/experimental\|melior_transformers/classification/transformer_models\|melior_transformers/custom_models

lint: clean
	flake8 melior_transformers tests \
		--exclude=melior_transformers/experimental,melior_transformers/classification/transformer_models,melior_transformers/custom_models
	black --check melior_transformers tests \
		--exclude melior_transformers/experimental\|melior_transformers/classification/transformer_models\|melior_transformers/custom_models

types:
	pytype --keep-going melior_transformers --exclude melior_transformers/experimental melior_transformers/classification/transformer_models melior_transformers/custom_models

test: clean
	pytest tests --cov melior_transformers/classification melior_transformers/ner melior_transformers/question_answering

# if this runs through we can be sure the readme is properly shown on pypi
check-readme:
	python setup.py check --restructuredtext
