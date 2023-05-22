# ####################################
# SpeechLM Base model #
# ####################################
[ $# -lt 3 ] && echo "Usage: $0 <model_path> <data_dir> <cpt_tag> <freeze_layers> <speech_subset> [mount=${PWD}] [world_size=8]" && exit 1
[ ${PWD##*/} != SpeechLM ] && echo "Error: dir not match! Switch to SpeechLM/ and run it again!" && exit 1

w2v_path=$1
DATA_DIR=$2
cpt=$3
freeze_layers=$4
speech_subset=$5
mount=$6
world_size=$7
[ -z $mount ] && mount=${PWD}
[ -z $world_size ] && world_size=1

CODE_ROOT=${PWD}

exp_name=${w2v_path%/*}
exp_name=${exp_name##*/}
MODEL_DIR="${mount}/exp/finetune_slue_ner/$exp_name/${cpt}"
[ -d $MODEL_DIR ] || mkdir -p $MODEL_DIR

config_dir=$CODE_ROOT/speechlm/config/slue/
config=speechlm_base_mixed_ner_finetune
train_subset="fine-tune-${speech_subset}"
valid_subset=dev

max_tokens=10000

python $CODE_ROOT/fairseq/fairseq_cli/hydra_train.py \
  --config-dir $config_dir \
  --config-name $config \
  common.user_dir=$CODE_ROOT/speechlm \
  \
  task.data=$DATA_DIR \
  task.text_data=$CODE_ROOT/dataset/slue-voxpopuli/phone_unit/bin-idx/ \
  model.w2v_path=${w2v_path} \
  model.freeze_layers="$freeze_layers" \
  \
  dataset.max_tokens=$max_tokens \
  distributed_training.distributed_world_size=${world_size} \
  \
  dataset.train_subset=$train_subset \
  dataset.valid_subset=$valid_subset \
  \
  common.tensorboard_logdir=$MODEL_DIR \
  checkpoint.save_dir=$MODEL_DIR \
  hydra.run.dir=$MODEL_DIR \
  hydra.job.name=${exp_name}

# model_path=/mnt/default/v-ziqzhang/data/speechulm/exp/base/base_speechlmp_32gpu_1accum/checkpoint_298_400000.pt
# data_dir=/home/v-ziqzhang/dataset/LibriSpeech/asr
