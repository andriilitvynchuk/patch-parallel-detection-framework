# Detection of pig-dogs military vehicles [![tests](https://github.com/yehor-morylov/drone_detection_inference_logic/actions/workflows/tests.yaml/badge.svg)](https://github.com/yehor-morylov/drone_detection_inference_logic/actions/workflows/tests.yaml)

## Using source code without cloning

You can install this repository as python package. To do this:
```
pip install git+https://github.com/yehor-morylov/drone_detection_inference_logic.git
```

## Preparing of environment

For development:
```
git clone https://github.com/yehor-morylov/drone_detection_inference_logic.git && cd drone_detection_inference_logic
pip install -e .
```

It will install all dependencies and will link `dronedet` to this folder (you don't need to reinstall it after every code change).
If you want to create **pull request**, you also need to do this before commits (once):
```
pip install -r requirements-dev.txt
pre-commit install
```
This will enable pre-commit hooks that will prettify your code and check for mypy, flake8 and other errors. To do force commits (e.g. you want to fix warnings later):
```
git commit -m "my message" -n
```
But don't use it too much - anyway each PR has CI for pre-commit checks and you will need to handle it :) You can use it just for emergency cases.