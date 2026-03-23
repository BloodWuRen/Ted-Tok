# Synthetic Task

## Task Definition

Each example follows a simple arithmetic format:

```
A+B=C
A-B=C
```

## Dataset Split

We simulate temporal drift by constructing two datasets:

| Dataset | Range of A, B	| Range of C | Size         |
|---------|---------------|------------|--------------|
| Old     | [0, 50]       | [0, 100]   | $\sim$ 3.9k  |
| New     | [0, 200]      | [0, 400]   | $\sim$ 60.7k |

Furthermore, we split the *New* dataset into a training set and a test set, with the test set containing 2000 examples.

## Model Architecture

We use a Transformer-based language model with 2 encoder layers, 4 attention heads, and embedding dimension 128 for next-token prediction. The model is intentionally small (~4K parameters) to isolate the effect of tokenizer differences.

## Data Generation

```
# Run the following command to generate the dataset.
python ./data_gen.py --n 50 --output_path ./data/
python ./data_gen.py --n 200 --output_path ./data/ --split
```

You can use the above commands to generate the dataset. The `--n` argument specifies the upper limit for numbers in combinations, and the `--output_path` argument specifies where to save the generated file. If you want to split the dataset into train and test sets, you can add the `--split` flag.

## Training

```
mkdir -p logs
python train.py --train_type D 2>&1 | tee logs/train_D.log
```

You can using the above command to train the model. The `--device` argument specifies the device to use for training. The `--checkpoint_path` argument specifies where to save model checkpoints. `--old_dataset_path`, `--new_dataset_path`, and `--test_dataset_path` specify the paths to the training and test datasets.

The `--train_type` argument specifies the training type, which can be either `D` for *Ted-Tok* baseline, `C` for *Old* baseline and `E` for *New* baseline. You can find more details about these training types in paper.

The model will be trained for 1000 epochs on both the new and old datasets. And report the accuracy on the test set.