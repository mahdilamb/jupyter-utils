.PHONY: help requirements install install-all qc test ruff mypy prune-branches
default: help

PACKAGE_DIR=temporary_python_project
SRC_FILES=${PACKAGE_DIR} tests

requirements: # Compile the pinned requirements if they've changed.
	@[ -f "requirements.in.md5" ] && md5sum --status -c requirements.in.md5 ||\
	( md5sum requirements.in > requirements.in.md5 && (python3 -c 'import piptools' || pip install pip-tools ) && pip-compile requirements.in -o requirements.txt )

install: # Install minimum required packages.
	@make requirements && pip install -e .${extras}

install-all: # Install all packages
	@make install extras='[all]'

ruff: # Run ruff
	@ruff check ${SRC_FILES} --fix

mypy: # Run mypy
	@mypy ${SRC_FILES}

test: # Run pytest
	@pytest --cov=${PACKAGE_DIR} tests --cov-report term-missing

qc:  # Format and test
	@make ruff; make mypy; make test

prune-branches: # Remove all branches except one
	@git branch | grep -v "${except}" | xargs git branch -D

prune-branches: except=main

help: # Show help for each of the Makefile recipes.
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | sort | while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m\n\t$$(echo $$l | cut -f 2- -d'#')\n"; done
