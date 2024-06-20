# Does Meta-learning Help mBERT for Few-shot Question Generation in a Cross-lingual Transfer Setting for Indic Languages?

### Requirements

- python 3.7
- pytorch 1.0.0
- pytorch_pretrained_bert



### Dataset:
https://github.com/google-research-datasets/tydiqa

### Run the code
For meta training:
Please create source language data in files named train_en.json and dev_en.json

Run python3 main.py --result_dir xyz

where xyz is your desired result directory

For zero-shot:
Please create target language data dev_ben.json for bengali and dev_tel.json for telugu

Run python3 main.py --zero_shot --model_dir models/xyz --test_langs ben

you can use any set of languages for test languages argument


For k-shot

Please create target language train data and dev data


Run python3 main.py --k_shot 32 --max_ft_steps 20 --model_dir models/en_tel --test_langs ben

you can change values of k, maximum fine tuning steps, model directory and names of test languages accordingly.
