#!/bin/bash

for n in {0..75}; do \
    python main.py -i 40 -t 3000 -k ${n}; \
done
