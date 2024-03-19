.PHONY: build clean develop install test uninstall

build:
	python setup.py build

develop:
	python setup.py develop

install:
	python setup.py install

uninstall:
	pip uninstall --yes tglite

test:
	pytest --cov=tglite

clean:
	rm -rf **/*/__pycache__ .pytest_cache
	rm -rf build dist **/*.egg-info
	rm -rf .coverage **/*/*.so
