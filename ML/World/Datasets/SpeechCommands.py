import torch

from torchaudio.datasets import SPEECHCOMMANDS


class SpeechCommands(SPEECHCOMMANDS):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.collate_fn = collate_fn

    def __getitem__(self, item):
        waveform, sample_rate, label, speaker_id, utterance_number = super().__getitem__(item)

        return waveform, label


def collate_fn(batch):
    waveform, label = zip(*batch)
    return pad_sequence(waveform), torch.stack(label)  # Pad waveform


def pad_sequence(batch: (list, tuple)):
    # Pad uneven sequences in a list with zeros up to the same max length
    return torch.nn.utils.rnn.pad_sequence([item.t() for item in batch],
                                           batch_first=True, padding_value=0.).permute(0, 2, 1)
