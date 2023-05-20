import torch
from fairseq.data import AddTargetDataset, data_utils

class AddSequentialLabelDataset(AddTargetDataset):
    def __init__(self, dataset, labels, **kwargs):
        super().__init__(dataset, labels, **kwargs)
    
    def size(self, index):
        return self.dataset.size(index)

    def collater(self, samples):
        collated = self.dataset.collater(samples)
        if len(collated) == 0:
            return collated
        target = [self.get_label(i, process_fn=self.process_label) for i in collated["id"]]

        if self.add_to_input:
            eos = torch.LongTensor([self.eos])
            prev_output_tokens = [torch.cat([eos, t], axis=-1) for t in target]
            target = [torch.cat([t, eos], axis=-1) for t in target]
            collated["net_input"]["prev_output_tokens"] = prev_output_tokens

        if self.batch_targets:
            collated["target_lengths"] = torch.LongTensor([len(t) for t in target])
            target = data_utils.collate_tokens(target, pad_idx=self.pad, left_pad=False)
            collated["ntokens"] = collated["target_lengths"].sum().item()
            if getattr(collated["net_input"], "prev_output_tokens", None):
                collated["net_input"]["prev_output_tokens"] = data_utils.collate_tokens(
                    collated["net_input"]["prev_output_tokens"],
                    pad_idx=self.pad,
                    left_pad=False,
                )
        else:
            collated["ntokens"] = sum([len(t) for t in target])

        collated["target"] = target
        return collated