#!/usr/bin/env bash



rm model.h5
python model.py
python drive.py model.h5