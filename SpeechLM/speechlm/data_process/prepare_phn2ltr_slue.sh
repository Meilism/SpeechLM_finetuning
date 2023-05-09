#!/bin/bash
[ ${PWD##*/} != SpeechLM ] && echo "Error: dir not match! Switch to SpeechLM/ and run it again!" && exit 1
data_dir=$1
dataset_name=$2
cwd=${PWD}
src=${PWD}/speechlm/data_process

set -e

for split in fine-tune dev; do
    mkdir -p dataset/${dataset_name}/phone_unit/tmp && cd dataset/${dataset_name}/phone_unit/

    echo "--------------------------------------------------------------------------------------"
    echo "--------Tokenize the text..."
    echo "--------------------------------------------------------------------------------------"
    cat ${data_dir}/${split}.wrd | python $src/wrd2ltr.py > tmp/${split}.ltr

    echo "--------------------------------------------------------------------------------------"
    echo "--------Tokenize the text to the kaldi-style phonemes ..."
    echo "--------------------------------------------------------------------------------------"
    python $src/phoneme_tokenizer/ltr2kaldi_phn_sil025.py -i tmp/${split}.ltr -o tmp/${split}
    cat tmp/${split}.kaldi_phn_sil025 | sed 's/SIL_S/SIL/g' > tmp/${split}.phn

    echo "--------------------------------------------------------------------------------------"
    echo "--------Filter too long samples and up-sample phonemes ..."
    echo "--------------------------------------------------------------------------------------"
    python $src/phoneme_tokenizer/repeat_withou_insert_sil_less_4375.py \
        tmp/${split}.phn \
        $src/phoneme_tokenizer/mean5_and_std25_sil14_spn32.dict \
        tmp/${split}_upsample.phn

    mv tmp/${split}_upsample.phn ${split}_upsample.phn-ltr.phn
    mv tmp/${split}.phn ${split}.phn-ltr.phn
    mv tmp/${split}.ltr ${split}.phn-ltr.ltr


    echo "--------------------------------------------------------------------------------------"
    echo "--------Create binary files ..."
    echo "--------------------------------------------------------------------------------------"
    [ ! -f bin-idx/dict.phn.txt ] && echo "dict ${cwd}/dataset/${dataset_name}/bin-idx/dict.phn.txt not found!" && exit 1
    [ ! -f bin-idx/dict.ltr.txt ] && echo "dict ${cwd}/dataset/${dataset_name}/bin-idx/dict.ltr.txt not found!" && exit 1
    bash $src/txt2idx.sh ${split}_upsample.phn-ltr.phn bin-idx bin-idx/dict.phn.txt
    bash $src/txt2idx.sh ${split}.phn-ltr.ltr bin-idx bin-idx/dict.ltr.txt

    rm -r tmp
    cd -
    echo "--------------------------------------------------------------------------------------"
    echo "--------Done! files are in ${PWD}/dataset/${dataset_name}"
    echo "--------------------------------------------------------------------------------------"
done
