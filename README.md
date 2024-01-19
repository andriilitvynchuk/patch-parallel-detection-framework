# Patch detection [![tests](https://github.com/litvinich/patch-parallel-detection-framework/actions/workflows/tests.yaml/badge.svg)](https://github.com/litvinich/patch-parallel-detection-framework/actions/workflows/tests.yaml)

## Using source code without cloning

You can install this repository as python package for code reuse. To do this:
```
pip install git+https://github.com/litvinich/patch-parallel-detection-framework.git
```
Still, you need install [shared_numpy](https://github.com/dillonalaird/shared_numpy.git) library and download the model (git lfs doesn't pull it with pip install). If you want to test it locally I recommend pulling the repository.

## Preparing of environment

For development:
```
git clone --recursive https://https://github.com/litvinich/patch-parallel-detection-framework.git && cd drone_detection_inference_logic
pip install -e .
cd shared_numpy && python setup.py build_ext --inplace && pip install -e . && cd ..
```
If you have installed git lfs after the pulling:
```
git lfs install
git lfs pull
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
