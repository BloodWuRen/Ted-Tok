# Machine Translation Task

## Task Definition

This directory contains the code for the machine translation (MT) experiment in our paper. The experiment studies how different tokenization strategies behave in a lifelong learning setting, where translation data is processed in temporal order.

## Model Architecture

We use a GPT-2 style Transformer model with 12 layers, 12 attention heads, hidden size 768, and context length 1024. The model size is approximately 124M parameters. We restrict the vocabulary size to 50,257.

## Data preparation

We downloaded the German–English parallel data for 2009–2019 from the WMT translation task download pages, using URLs of the form https://www.statmt.org/wmtXX/translation-task.html#download, where XX ranges from 09 to 19. Since each year’s release may include data from previous years, we deduplicated the corpus and assigned each sample to the first year in which it appeared. Because the data volume for 2014–2016 is relatively small, we merged the 2014–2017 data into a single time period, resulting in 8 final time segments.

| Year Range | # Training Examples |
|------------|---------------------|
| 2009       | 1.46 M              |
| 2010       | 0.24 M              |
| 2011       | 0.22 M              |
| 2012       | 0.19 M              |
| 2013       | 2.41 M              |
| 2014-2017  | 1.11 M              |
| 2018       | 26.75 M             |
| 2019       | 34.78 M             |

We will release the preprocessed data in the future.

## Training

