# A Reverse Approach to Solving Reading Comprehension and Logical Reasoning Tasks

(INCOMPLETE)

A Reverse Approach to Solving Reading Comprehension and Logical Reasoning Tasks. Jeffrey Lu, Ivan Rodriguez. 2022.

[Paper Link](https://link.com)

## Project Structure

```
|--- dataset
|--- models
|--- trainers
|--- results
|--- util
```

## Pretrained Models

Our pretrained mdoels are from Huggingface Transformers. We used the following models:
- ALBERT (albert-large-v2)
- DistilBert (distilbert-base-cased)
- RoBERTa (distilroberta-base)
- Bart (facebook/bart-base)

## Training

All trainers are in the `trainers` directory and can be run by executing

```
python3 trainers/(trainer_file_name).py
```
