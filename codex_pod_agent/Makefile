SHELL := /bin/bash

.PHONY: dev build test lint run

dev:
	bash scripts/dev.sh

build:
	bash scripts/build.sh

test:
	bash scripts/test.sh

lint:
	bash scripts/lint.sh

run:
	python -m podagent.runner --once

