import copy
import torch
from torch.utils.data import DataLoader


class TorchDataloader:

    def __init__(self, dataloader):
        self._dataloader = dataloader

    def __iter__(self):
        return iter(self._dataloader)

    def __len__(self):
        return len(self._dataloader)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        create_config = copy.deepcopy(config)
        batch_processor = BasicBatchProcessor()
        return cls(dataloader=DataLoader(kwargs['dataset'], collate_fn=batch_processor, **create_config))


class BasicBatchProcessor:

    def __call__(self, batch):
        processed_batch = {}

        for key in batch[0].keys():
            if key.endswith('.ids'):
                prefix = key.split('.')[0]
                assert '{}.length'.format(prefix) in batch[0]

                processed_batch[f'{prefix}.ids'] = []
                processed_batch[f'{prefix}.length'] = []

                for sample in batch:
                    processed_batch[f'{prefix}.ids'].extend(sample[f'{prefix}.ids'])
                    processed_batch[f'{prefix}.length'].append(sample[f'{prefix}.length'])

        for part, values in processed_batch.items():
            processed_batch[part] = torch.tensor(values, dtype=torch.long)

        return processed_batch
