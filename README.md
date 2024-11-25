# 25422_tls_bert
Fine-tuning pre-trained BERT model for detecting server TLS misconfiguration.

## Dependencies

Python 3.7+

## Setup

#### Clone Repo
```bash
git clone https://github.com/andrewcchu/25422_tls_bert.git
cd 25422_tls_bert
```

#### Setup Python Virtual Environment
NOTE: If different/needed, replace `python` with whatever your Python 3.7+ is aliased to (e.g., python3, python3.11, etc.)
```bash
python -m pip install virtualenv
virtualenv -p python venv
source venv/bin/activate
```

#### Install required packages
```bash
python -m pip install -r REQUIREMENTS.txt
```

#### Get model files
```bash
wget -O model.zip "https://uchicagoedu-my.sharepoint.com/:u:/g/personal/andrewcchu_uchicago_edu/EWmZeAEPBwxJnFpzUeGssRQBlCwldMsK9-8xGB77_u0ArQ?e=KHdfi6&download=1"
unzip model.zip
```

## Fine-Tuning

#### Data Description

The data in this repo represents parsed values of the header fields of packets in the TLS handshake sent back to the client by websites in the Tranco top list. Feel free to look at its format in `/dataset`. There are two subfolders here: `binary` and `top_multi`. `binary` labels all parsed handshakes as coming from either a "Properly Configured" or "Misconfigured" server. `top_multi` labels all "Misconfigured" handshakes from the binary set with a more specific reason for misconfiguration. Websites and labels for these datasets were determined using [`testssl.sh`](https://github.com/drwetter/testssl.sh).

#### Running Fine-tuning

Fine-tuning can be run as follows (again, replacing `python3.7` with whatever Python you installed the dependencies above with):
```bash
python multi_classifier.py \
--task_name=cola \
--do_train=true \
--do_eval=true \
--data_dir=./dataset/top_multi \
--vocab_file=./model/vocab.txt \
--bert_config_file=./model/bert_config.json \
--init_checkpoint=./model/model.ckpt-1000000 \
--max_seq_length=128 \
--train_batch_size=8 \
--learning_rate=2e-5 \
--num_train_epochs=10.0 \
--output_dir=./output/multi/top/top_multi_10_epochs_2e5_lr_128_max_8_bs_uncased \
--do_lower_case=True \
--save_checkpoints_steps 1000
```

Breaking down above:
1. `--task_name=cola`: Specifies the pre-trained model to use the [Corpus of Linguistic Acceptability (CoLA)](https://arxiv.org/pdf/1805.12471.pdf) task. For our purposes, this just tells the model to run/train on a classification task
2. `--do_train=true`: Specifies to train (i.e., fine-tune since we are using a base model) the base model
3. `--do_eval=true`: Perform evaluation after fine-tuning to evaluate outcome
4. `--data_dir=./dataset/top_multi`: Specifies fine-tuning data directory
5. `--vocab_file=./model/vocab.txt`: Vocab file for model tokens
6. `--bert_config_file=./model/bert_config.json`: Specifies the config for the model
7. `--init_checkpoint=./model/model.ckpt-1000000`: Specifies the base, pre-trained model
8. `--max_seq_length=128`: Specifies the longest input for use in training from the dataset in `--data_dir`. If an input is longer than `--max_seq_length`, it will be truncated
9. `--train_batch_size=8`: Specifies how many inputs to allow the model to process at a time. NOTE: this value is highly tied to resources
10. `--learning_rate=2e-5`: Specifies the learning rate (i.e., how much to change the model in response to the estimated error each time the model weights are updated)
11. `--num_train_epochs=10.0`: Specifies the number of epochs (i.e., number of full passes over all data) for the model
12. `--output_dir=./output/multi/top/top_multi_10_epochs_2e5_lr_128_max_32_bs_uncased`: Specifies output directory; note the format of the output folder name, following the above parameters
13. `--do_lower_case=True`: Specifies if input should be preprocessed to lowercase for the model 
14. `--save_checkpoints_steps 10`: After how many training iterations a checkpoint should be saved

The fine-tuning process will take a while, especially if you do not have a GPU. After finishing, check the `output` directory and its subfolders for the output of the fine-tuning process (model checkpoint(s), eval results).
