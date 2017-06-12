#!/bin/bash

for n in {48..74}; do \
    python main.py -i 40 -t 3000 -k ${n}; \
done
