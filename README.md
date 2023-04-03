# Polytuplet Loss: A Reverse Approach to Training Reading Comprehension and Logical Reasoning Models

Jeffrey Lu, Ivan Rodriguez. 2023.

[Paper Link Pending]

## Project Structure

```
|--- dataset
    |--- raw_data
    |--- cleaned
|--- preprocess
    |--- preprocess.py
    |--- run_cleaning.py
|--- models
tuner.py
README.md
```

## Preprocessing
Text data is minimally cleaned before being reshaped and tokenized appropriately for the various pretrained models.

Data is either mixed or unmixed. Mixed data preserves the ordering of answers in the raw data. Unmixed data minimally reorders answers such that the correct answer is always presented last. Both mixed and unmixed data is generated for each configuration.

Cleaning can be completed by executing the following command:

```
python preprocess/run_cleaning.py
```

Final preprocessing is completed by the trainer.

## Pretrained Models

Our pretrained models are from [Huggingface Transformers](https://huggingface.co/models). We used the following models:
- ALBERT (albert-base-v2, albert-xxlarge-v2)
- RoBERTa (roberta-large)
- BERT (bert-base-uncased)
- DistilBert (distilbert-base-uncased)

Note that BERT training was completed in Google Colaboratory, and is therefore not covered in this repository. BERT training and tuning can be easily added by importing and configuring the BERT model from Huggingface using the same method that all the other models were imported and configured.

## Training

Our training was completed on the TPU configuration available in Google Colab (8 TPU v2) and in the TPU v3-8 VMs available from the Google TPU Research Cloud (TRC) program. This codebase may be easily adjusted to run on non-TPU machines.

Our results were extracted directly from the tuner. The tuner can be run using the following command, replacing items in brackets with the appropriate choices:

```
python tuner.py [baseline|polytuplet] [0|1|2] [mixed|unmixed]
```
