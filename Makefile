export PIPENV_VENV_IN_PROJECT := 1
export PIPENV_VERBOSITY := -1

environment:
	@echo "Building Python environment"
	python3 -m pip install --upgrade pip
	pip install --upgrade pipenv
	pipenv install --python 3.9

start_env:
	@echo "Starting Python environment (version 3.9.17)"
	pipenv shell

start_jupyter:
	pipenv run jupyter notebook

clean:
	@echo "Cleaning"
	pipenv --rm






code:
	@echo "Code formatting with black, isort, and pylint"
	black .
	isort .
	pylint --recursive=y .