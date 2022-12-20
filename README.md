# A Reverse Approach to Solving Reading Comprehension and Logical Reasoning Tasks

(INCOMPLETE)

A Reverse Approach to Solving Reading Comprehension and Logical Reasoning Tasks. Jeffrey Lu, Ivan Rodriguez. 2022.

[Paper Link](https://link.com)

## Project Structure

```
|--- dataset
    |--- raw_data
    |--- cleaned
    |--- processed
|--- preprocess
    |--- preprocess.py
    |--- run_cleaning.py
    |--- run_preprocess.py
|--- models
|--- trainers
|--- results
|--- util
```

## Preprocessing
Text data is minimally cleaned before being reshaped and tokenized appropriately for the various pretrained models.

Cleaning and the rest of preprocessing can be completed by executing the following commands in order:

```
python preprocess/run_cleaning.py
python preprocess/run_preprocess.py
```

## Pretrained Models

Our pretrained models are from Huggingface Transformers. We used the following models:
- ALBERT (albert-large-v2)
- DistilBert (distilbert-base-cased)
- RoBERTa (distilroberta-base)
- Bart (facebook/bart-base)

## Training

All trainers are in the `trainers` directory and can be run by executing

```
python trainers/(trainer_file_name).py
```
