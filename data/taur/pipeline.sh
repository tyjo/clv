#!/usr/bin/env bash

python convert_to_otu_table.py
python filter.py
python get_top_otus.py
python make_event_table.py
