#!/bin/bash

./scripts/jupytext_convert.sh neurogolf --to-ipynb --destination notebooks --exclude __init__.py --maintain-structure
