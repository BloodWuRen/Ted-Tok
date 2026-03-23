# QA Task

## Task Definition

The QA task is based on the [StreamingQA](https://github.com/google-deepmind/streamingqa) dataset, which consists of temporally evolving question–answer pairs derived from news articles.

The objective is to generate an answer given a question, and evaluate whether the generated output contains the correct answer.

## Dataset Split

We simulate temporal drift using time-ordered data from news articles spanning multiple years.

The dataset is grouped into consecutive time intervals:

| Time Range   | # Training Examples |
|--------------|--------------------|
| 2007–2009    | 6.08M              |
| 2010–2012    | 7.20M              |
| 2013–2015    | 12.49M             |
| 2016–2018    | 10.76M             |
| 2019–2021    | 15.88M             |

The model is trained sequentially on these temporally ordered subsets to simulate a lifelong learning setting.

## Model Architecture

We use a GPT-2 style Transformer model with 12 layers, 12 attention heads, hidden size 768, and context length 1024. The model size is approximately 124M parameters.

## Dataset preparation

```
# Run the following command to download the dataset and prepare it for training.
python ./data_processer/datasets_download.py --data_path ./data/WMT

./data_processer/prepare_train_data.sh --data_path ./data/WMT --output_path ./data/WMT_rawtext
```

The full dataset will be downloaded in `data_path`. To simulate temporal drift, datasets are partitioned into yearly subsets, stored in `output_path`. 

```
# Run the following command to build the tokenizers for the dataset.
./data_processer/build_tokenizers.sh --data_path ./data/WMT --output_raw_path ./data/WMT_rawtext --output_tokenizer_path ./checkpoints/tokenizers --start_year 2007 --end_year 2009
```

Build a tokenizer for the dataset from `start_year` to `end_year` using BPE, and save it in the `output_tokenizer_path` directory with the name `custom_gpt2_tokenizer_{start_year}-{end_year}`.

## Training

There are commands for training the model according to the *Old*, *New* and *Ted-Tok* baselines. You can find more details about these training types in the paper. 

```
# Run the following command to train the model according to the *Old* baseline.
./config/cpt.sh --data_path ./data/WMT_rawtext --tokenizers_path ./checkpoints/tokenizers --output_path ./checkpoints/models --tok_start_year 2007 --tok_end_year 2009 --train_start_year 2007 --train_end_year 2021
```

Train the model on the dataset from `train_start_year` to `train_end_year` using the tokenizer built for the dataset from `tok_start_year` to `tok_end_year`. The trained model will be saved in the `output_path` directory. Before training, you must build the tokenizer for the dataset from `tok_start_year` to `tok_end_year` as described in the previous section.

```
# Run the following command to train the model according to the *Ted-Tok* baseline.
./config/cpt.sh --data_path ./data/WMT_rawtext --tokenizers_path ./checkpoints/tokenizers --output_path ./checkpoints/models --tok_start_year 2007 --tok_end_year 2009 --train_start_year 2007 --train_end_year 2018

./config/ted-tok.sh --data_path ./data/WMT_rawtext/wmt_2019-2021 --prev_out_dir ./checkpoints/models/ours-small-10epoch_finetune_wmt_2016-2018_tok2007-2009 --tokenizers_path ./checkpoints/tokenizers --output_path ./checkpoints/models --tok_start_year 2007 --tok_end_year 2009
```

If you have already trained the model according to the *Old* baseline, you can only run the second command.

```
# Run the following command to train the model according to the *New* baseline.
./config/cpt.sh --data_path ./data/WMT_rawtext --tokenizers_path ./checkpoints/tokenizers --output_path ./checkpoints/models --tok_start_year 2007 --tok_end_year 2009 --train_start_year 2007 --train_end_year 2018

./config/change_new_tok.sh --data_path ./data/WMT_rawtext/wmt_2019-2021 --prev_out_dir ./checkpoints/models/ours-small-10epoch_finetune_wmt_2016-2018_tok2007-2009 --tokenizers_path ./checkpoints/tokenizers --output_path ./checkpoints/models --tok_start_year 2019 --tok_end_year 2021
```

If you have already trained the model according to the *Old* baseline, you can only run the second command. Before training, you must build the tokenizer for the dataset from `tok_start_year` to `tok_end_year` as described in the previous section.

## Evaluation

```
# Run the following command to transform the checkpoint to Hugging Face format.
python ./nanoGPT/ckpt2hf.py --checkpoint_path checkpoints/models/ours-small-10epoch_scratch_wmt_2007-2009_tok2007-2009 --iteration_number 200 --output_directory checkpoints/models/hf_ours-small-10epoch_scratch_wmt_2007-2009_tok2007-2009_iter200
```