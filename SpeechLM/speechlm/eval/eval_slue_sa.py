import numpy as np
import torch, argparse, json, os
from tqdm import tqdm
from torch import cuda
from fairseq.dataclass import FairseqDataclass
from speechlm.models.speechlm_cls import SpeechLMSeqCls
from speechlm.models.speechlm import SpeechlmModel
from speechlm.tasks import sequence_classification, joint_sc2t_pretrain
from sklearn.metrics import f1_score, precision_score, recall_score


def main():
    parser = argparse.ArgumentParser(
        description="Get evaluation result for sentiment analysis task",
    )
    parser.add_argument(
        "--data", type=str, required=True, help="manifest dir for data loading"
    )
    parser.add_argument(
        "--text_data", type=str, required=False, help="bin-idx/ directory which contains preprocessed dataset, only used for text inputs"
    )
    parser.add_argument(
        "--input", type=str, required=True, choices=["speech", "text"], help="speech or text inputs"
    )
    parser.add_argument(
        "--subset", type=str, required=True, help="split name (test, dev, finetune"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="save dir containing checkpoints folder",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        required=False,
        default="checkpoint_best.pt",
        help=".pt file you want to use",
    )
    parser.add_argument(
        "--use-gpu",
        default=False,
        action="store_true",
        help="use gpu if available default: False",
    )
    parser.add_argument(
        "--eval",
        default=True,
        action="store_true",
        help="eval after inference. save in the same filename with output but .json format. default: True",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        required=False,
        default="",
        help="suffix to add to the end of the output files",
    )
    args = parser.parse_args()
    print(args)

    if args.use_gpu:
        device = "cuda:0" if cuda.is_available() else "cpu"
    else:
        device = "cpu"
    checkpoint_dir = args.save_dir
    checkpoint = SpeechLMSeqCls.from_pretrained(
        checkpoint_dir, checkpoint_file=args.checkpoint_file
    )
    
    if args.input == "text":
        assert args.text_data is not None
    
    if args.input == "speech":
        task_cfg = checkpoint.task.cfg
        task_cfg.data = args.data
        task = sequence_classification.AudioClassificationTask(task_cfg)
    elif args.input == "text":
        if checkpoint.task.cfg['_name'] == 'slue_audio_classification':
            task_cfg = sequence_classification.TextClassificationConfig(**checkpoint.task.cfg)
        else:
            task_cfg = checkpoint.task.cfg
        task_cfg.data = args.data
        task_cfg.text_data = args.text_data
        task = sequence_classification.TextClassificationTask(task_cfg)
    task.load_dataset(args.subset)
    task.load_label2id
    checkpoint.to(device)
    data = task.datasets[args.subset]
    model = checkpoint.models[0]
    model.eval()
    preds = []
    gt = []

    with torch.no_grad():
        for iter in tqdm(range(len(data))):
            input = data.__getitem__(iter)
            output = model(
                source=input["source"].unsqueeze(0).to(device), padding_mask=None
            )
            preds.append(np.argmax(output["pooled"].cpu().numpy()))
            gt.append(input["label"])
    id2label = {i: l for i, l in enumerate(checkpoint.task.label2id)}

    if args.suffix:
        args.suffix = f"-{args.suffix}"
    output_tsv = os.path.join(args.save_dir, f"pred-{args.subset}-{args.input}{args.suffix}-input.sent")
    fid = open(output_tsv, "w")
    for sent_id in preds:
        fid.write(f"{id2label[sent_id]}\n")
    fid.close()

    if args.eval:
        output_json = os.path.splitext(output_tsv)[0] + ".json"
        json_dict = {}
        json_dict["macro"] = {
            "precision": precision_score(gt, preds, average="macro") * 100,
            "recall": recall_score(gt, preds, average="macro") * 100,
            "f1": f1_score(gt, preds, average="macro") * 100,
        }
        json_dict["micro"] = {
            "precision": precision_score(gt, preds, average="weighted") * 100,
            "recall": recall_score(gt, preds, average="weighted") * 100,
            "f1": f1_score(gt, preds, average="weighted") * 100,
        }
        json_dict["per_classes"] = {
            id2label[idx]: score
            for idx, score in enumerate(f1_score(gt, preds, average=None) * 100)
        }

        with open(output_json, "w") as fp:
            json.dump(json_dict, fp, sort_keys=True, indent=4)


if __name__ == "__main__":
    main()
