# Machine Translation Task

## Task Definition

This directory contains the code for the machine translation (MT) experiment in our paper. The experiment studies how different tokenization strategies behave in a lifelong learning setting, where translation data is processed in temporal order.

## Model Architecture

We use a GPT-2 style Transformer model with 12 layers, 12 attention heads, hidden size 768, and context length 1024. The model size is approximately 124M parameters. We restrict the vocabulary size to 50,257.

## Data preparation

We downloaded the English–German parallel data for 2009–2019 from the WMT translation task download pages, using URLs of the form https://www.statmt.org/wmtXX/translation-task.html#download, where XX ranges from 09 to 19. Since each year’s release may include data from previous years, we deduplicated the corpus and assigned each sample to the first year in which it appeared. Because the data volume for 2014–2016 is relatively small, we merged the 2014–2017 data into a single time period, resulting in 8 final time segments.

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

We process the WMT English-German parallel corpus following the standard WMT preprocessing and tokenization pipeline to ensure comparability with prior work.

```
# Run the following command to build the tokenizer.
cd MT
./config/build_tokenizer.sh --input_dir ./data/wmt09/de-en/wmt09_en_de_decoder_only_dataset/train/train.parquet --save_dir ./checkpoints/tokenizers/custom_tokenizer_wmt09
```

`input_dir` specifies the path to the training dataset, and `save_dir` specifies where to save the trained tokenizer.

## Training

```
# Run the following command to train the model.
python ./config/model_init.py --tokenizer_path ./checkpoints/tokenizers/custom_tokenizer_wmt09 --save_path ./checkpoints/models/09tok_en-cs_from_scratch_gpt2
./config/09tok_wmt09-18_3epoch.sh
```

These commands will train a model on the 2009–2018 data, which will be used for different variants of continue-pretraining on the 2019 data.

```
# Run the following command to continue-pretrain the model according to the *Old* baseline.
./config/09tok_wmt09-19_3epoch.sh
```

```
# Run the following command to continue-pretrain the model according to the *Ted-Tok* baseline.
./config/09tok-wmt09-19_tedtok_3epoch_numupdate100.sh
```

If you want to continue pretraining the model according to the *New* baseline, you need to manually modify the tokenizer of the model trained on the 2009–2018 data, which is saved in the `./checkpoints/models/09tok_wmt18_3epoch` directory by default. Then you can directly use the Old baseline command to continue pretraining.