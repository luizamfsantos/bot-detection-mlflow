#!/bin/bash
curl -L -o data/raw/users_vs_bots.zip https://www.kaggle.com/api/v1/datasets/download/juice0lover/users-vs-bots-classification
unzip -o data/raw/users_vs_bots.zip -d data/raw/
rm data/raw/users_vs_bots.zip
