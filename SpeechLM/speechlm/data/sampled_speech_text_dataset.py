from typing import Callable, Dict, List
from fairseq.data import MultiCorpusSampledDataset, FairseqDataset

class SampledSpeechTextDataset(MultiCorpusSampledDataset):
    def __init__(
        self,
        datasets: Dict[str, FairseqDataset],
        sampling_func: Callable[[List], int] = None,
    ):
        super().__init__(datasets, sampling_func)

    def num_tokens(self, index: int):
        """
        Return an example's length (number of tokens), used for batching. We 
        divide the shape of speech signals by 16000/50 = 320 to match the 
        resolution of text tokens and return the max across all examples at
        index across speech and text datasets.
        """
        n_tokens_0 = self.datasets[0].num_tokens(self._map_index_to_dataset(0, index))
        n_tokens_1 = self.datasets[1].num_tokens(self._map_index_to_dataset(1, index))
        return max(
            n_tokens_0[0] / 320 if isinstance(n_tokens_0, tuple) else n_tokens_0 / 320,
            n_tokens_1[0] / 320 if isinstance(n_tokens_1, tuple) else n_tokens_1,
        )

    def size(self, index: int):
        """
        Return an example's size as a float or tuple. We divide the shape
        of speech signals by 16000/50 = 3200 to match the resolution of 
        text tokens.
        """
        size_0 = self.datasets[0].size(self._map_index_to_dataset(0, index)) 
        size_1 = self.datasets[1].size(self._map_index_to_dataset(1, index)) 
        return max(
            size_0[0] / 320 if isinstance(size_0, tuple) else size_0 / 320,
            size_1[0] / 320 if isinstance(size_1, tuple) else size_1,
        )