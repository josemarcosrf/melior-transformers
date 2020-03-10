help:
	@echo "    install"
	@echo "        Install core dependencies."
	@echo "    upload-package"
	@echo "        Upload package to Melior Pypi server"
	@echo "    clean"
	@echo "        Remove Python/build artifacts."
	@echo "    formatter"
	@echo "        Apply black formatting to code."
	@echo "    lint"
	@echo "        Lint code with flake8, and check if black formatter should be applied."
	@echo "    types"
	@echo "        Check for type errors using pytype."
	@echo "    test"
	@echo "        Run pytest on tests/."
	@echo "    check-readme"
	@echo "        Check if the README can be converted from .md to .rst for PyPI."

install:
	pip install -e .[dev]
	pip list

upload-package: clean
	python setup.py sdist
	twine upload dist/* -r melior

clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f  {} +
	rm -f .coverage*
	rm -rf outputs
	rm -rf cache_dir
	rm -rf runs
	rm -rf data

formatter:
	black melior_transformers tests \
		--exclude melior_transformers/experimental\|melior_transformers/classification/transformer_models\|melior_transformers/custom_models
	isort --recursive melior_transformers examples tests
		

lint: clean
	flake8 melior_transformers tests \
		--exclude=melior_transformers/experimental,melior_transformers/classification/transformer_models,melior_transformers/custom_models
	black --check melior_transformers tests \
		--exclude melior_transformers/experimental\|melior_transformers/classification/transformer_models\|melior_transformers/custom_models
	isort --recursive melior_transformers examples tests --check
	

types:
	pytype --keep-going melior_transformers --exclude melior_transformers/experimental melior_transformers/classification/transformer_models melior_transformers/custom_models

test: clean
	pytest tests --cov melior_transformers/classification melior_transformers/ner melior_transformers/question_answering

# if this runs through we can be sure the readme is properly shown on pypi
check-readme:
	python setup.py check --restructuredtext
