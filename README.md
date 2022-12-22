# A Reverse Approach to Solving Reading Comprehension and Logical Reasoning Tasks

(INCOMPLETE)

Jeffrey Lu, Ivan Rodriguez. 2022.

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

Data is either mixed or unmixed. Mixed data preserves the ordering of answers in the raw data. Unmixed data minimally reorders answers such that the correct answer is always presented last. Both mixed and unmixed data is generated for each configuration.

Cleaning and the other preprocessing can be completed by executing the following commands in order:

```
python preprocess/run_cleaning.py
python preprocess/run_preprocess.py
```

## Pretrained Models

Our pretrained models are from [Huggingface Transformers](https://huggingface.co/transformers/v3.3.1/pretrained_models.html). We used the following models:
- ALBERT (albert-large-v2)
- DistilBert (distilbert-base-cased)
- RoBERTa (distilroberta-base)
- Bart (facebook/bart-base)

## Training

Our training was completed on the TPU configuration available in Google Colab(8 TPU v2). This codebase may be easily adjusted to run on non-TPU machines.

All trainers are in the `trainers` directory and can be run by executing

```
python trainers/(trainer_file_name).py
```
