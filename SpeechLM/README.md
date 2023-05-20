
# TTIC 31120 2023 Spring Final Project: Fine-Tuning SpeechLM with Text

## Reproduce SpeechLM ASR Results
### Download LibriSpeech datasets
Download LibriSpeech train-clean-100 and dev-other subsets from https://www.openslr.org/12 and extract the files

### Install Fairseq
In the ``SpeechLM_finetuning/SpeechLM`` folder, run
``` bash
git clone https://github.com/facebookresearch/fairseq.git
cd fairseq/
git checkout 0338cdc3094ca7d29ff4d36d64791f7b4e4b5e6e
```

And remember to put the addons into fairseq directory before installing fairseq
``` bash
cp -r ../fairseq_addons/fairseq/* ./
pip3 install --editable .
```
If you fail to build the package, try to update your gcc version to gcc 5.4

### LibriSpeech ASR Data Preprocessing with Fairseq
In the ``SpeechLM_finetuning/SpeechLM`` folder, run
``` bash
mkdir manifest
mkdir manifest/librispeech/
dest_dir=path/to/manifest/librispeech/
cp dataset/LibriLM/phone_unit/bin-idx/dict.ltr.txt $dest_dir/
```
Then, in the ``SpeechLM_finetuning/SpeechLM/fairseq/`` folder, run the following cammands for both dev-other and train-clean-100
``` bash
python examples/wav2vec/wav2vec_manifest.py /path/to/libri/dev-other/ --dest $dest_dir --ext "flac" --valid-percent 0 
python examples/wav2vec/libri_labels.py $dest_dir/fine-tune.tsv --output-dir $dest_dir --output-name=dev-other                      
```
Then you should get the following files in your ``$dest_dir`` directory
- dev_other.ltr
- dev_other.tsv
- dev_other.wrd
- train_clean_100.ltr
- train_clean_100.tsv
- train_clean_100.wrd

### Fine-Tune SpeechLM on LibriSpeech ASR
First, reach out to me to get the pre-trained SpeechLM_P_Base checkpoint. Then do some modification to the checkpoint file
```python
state = torch.load(ckpt_pth)
state['cfg']['task']['text_cfg']['text_data'] = '/path/to/SpeechLM_finetuning/SpeechLM/dataset/LibriLM/phone_unit/bin-idx/'
state["cfg"]["model"]["freeze_layers"] = 0 
torch.save(state, ckpt_pth)
```

You're ready to train your ASR model! In ``SpeechLM_finetuning/SpeechLM/``, run
``` bash
model_path=path/to/your/pre-trained/model
data_dir=${dest_dir}
exp_name=YOUR_EXP_NAME
bash speechlm/scripts/tune_speechlm_asr/finetune_base_ctc.sh $model_path $data_dir $exp_name
```

### Fine-Tune SpeechLM on Slue Sentiment Analysis
Firstly, clone the Github repository of SLUE and download the dataset
``` bash
cd slue/directory
bash scripts/download_datasets.sh
```

Then go back to ``SpeechLM_finetuning/SpeechLM/``, and run the following command to fine-tune SpeechLM on SLUE sentiment analysis:
``` bash
freeze_layers=number/of/frozen/layers/between/0~6
slue_path=path/to/slue/manifest
bash speechlm/scripts/slue/finetune_base_speech_sa.sh $model_path ${slue_path}/slue-voxceleb/ $exp_name $freeze_layers
```
After fine-tuning the model, you will get fine-tuned checkpoints in ``exp/finetune_slue_sa/ckpt/${exp_name}``.
Before proceeding on, we need to a dirty trick to the checkpoint:
```python
state = torch.load(finetuned_ckpt_pth)
state['cfg']['model']['w2v_args'] = None
torch.save(state, finetuned_ckpt_pth)
```

Then, you are ready to get the sentiment analysis results! The last step is:
``` bash
save_dir=exp/finetune_slue_sa/ckpt/${exp_name}
python3 -m speechlm.eval.eval_slue_speech_sa --data $slue_path --subset dev --save-dir $save_dir  --checkpoint-file checkpoint_best.pt --use-gpu --eval
```
The results will also be put in ``$save_dir/``.

### Fine-Tune SpeechLM on Slue Sentiment Analysis with Text Inputs
First prepare text inputs for SpeechLM
``` bash
slue_data_dir=path/to/slue/manifest
bash speechlm/data_process/prepare_phn2ltr_slue.sh ${slue_data_dir}/slue-voxceleb/ slue-voxceleb
bash speechlm/data_process/prepare_phn2ltr_slue.sh ${slue_data_dir}/slue-voxpopuli/ slue-voxpopuli 
```

Then run the code below to fine-tune SpeechLM on text inputs
``` bash
freeze_layers=number/of/frozen/layers/between/0~6
bash speechlm/scripts/slue/finetune_base_text_sa.sh $model_path ${slue_data_dir}/slue-voxceleb/ $exp_name $freeze_layers
```

The fine-tuned checkpoint will be save in ``exp/finetune_slue_sa/ckpt/${exp_name}``.
Again, we need to modify the checkpoint by:
```python
state = torch.load(finetuned_ckpt_pth)
state['cfg']['model']['w2v_args'] = None
torch.save(state, finetuned_ckpt_pth)
```

Since the model is fine-tuned on text, we can evaluate it with either text or speech inputs. The corresponding commands are:
``` bash
save_dir=exp/finetune_slue_sa/ckpt/${exp_name}
input_type=speech/or/text
text_data=path/to/the/bin-idx/dir # only used when input_type=text
python3 -m speechlm.eval.eval_slue_sa --data $data_dir --input $input_type --subset dev --save-dir $save_dir  --checkpoint-file checkpoint_best.pt --use-gpu --eval [--text_data $text_data]
```

### Fine-Tune SpeechLM on Slue Named Entity Recognition with Speech or Text Inputs
To fine-tune the model, run
```bash
bash speechlm/scripts/slue/finetune_base_speech_ner.sh $model_path ${slue_data_dir}/slue-voxpopuli/e2e_ner/ $exp_name $freeze_layers
```
or
```bash
bash speechlm/scripts/slue/finetune_base_text_ner.sh $model_path ${slue_data_dir}/slue-voxpopuli/e2e_ner/ $exp_name $freeze_layers
```
for speech and text fine-tuning, respectively.

No matter the model is fine-tuned on speech or text, do the same trick again:
```python
state = torch.load(finetuned_ckpt_pth)
state['cfg']['model']['w2v_args'] = None
torch.save(state, finetuned_ckpt_pth)
```

Then we can run ASR decoding to get the NER results!
This step requires some additional packages, including [flashlight](https://github.com/flashlight/flashlight) and [kenlm](https://github.com/kpu/kenlm). 
Make sure you have all dependencies set up before executing the following lines:
```bash
input_type=speech/or/text
lm=nolm-argmax #"nolm-argmax" for argmax decoding, and "vp_ner/4" for decoding with kenlm
lmdir=path/to/your/4gram/lm/ #only used in kenlm decoding
max_tokens=4000000 #4000000 for speech inputs, and 10000 for text inputs
python3 -m speechlm.eval.eval_slue_asr eval_ctc_model --data $slue_data_dir --subset dev --model $save_dir --lm $lm --user-dir ${PWD}/speechlm --max_tokens $max_tokens --input_type $input_type --text_data $text_data --lmdir $lmdir
```

In the end, compute f1 score by running
```bash
python3 -m speechlm.eval.eval_slue_ner eval_ner --model_dir $save_dir --input_type $input_type --eval_set dev --eval_label combined --lm $lm
```