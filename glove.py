import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
from torchtext.vocab import GloVe
from homura.utils import Trainer, callbacks, reporter
from models import Encoder, Decoder


def get_device(module):
    return next(module.parameters()).device


class CTrainer(Trainer):
    def iteration(self, data, is_train):
        input, = self.to_device(data)
        output = self.model(input)
        loss = self.loss_f(output, input)
        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss, output


class Code(object):
    def __init__(self, encoder, vocab):
        self.encoder = encoder
        self.vocab = vocab

    def get_code(self, words):
        ids = [self.vocab.stoi[w] for w in words]
        v = self.vocab.vectors[ids].to(get_device(self.encoder))
        return self.encoder.get_code(v)


def main(epochs, batch_size, num_component, num_codevec, words):
    emb_dim = 300
    glove = GloVe(name="6B", dim=emb_dim)
    loader = DataLoader(TensorDataset(glove.vectors), batch_size=batch_size)
    compressor = nn.Sequential(Encoder(emb_dim, num_component, num_codevec),
                               Decoder(emb_dim, num_component, num_codevec))
    optimizer = torch.optim.Adam(compressor.parameters(), lr=1e-4)

    c = callbacks.CallbackList(callbacks.LossCallback(),
                               callbacks.WeightSave("checkpoints"),
                               callbacks.ParameterReporterCallback(reporter.TensorBoardReporter()))

    trainer = CTrainer(compressor, optimizer, F.mse_loss, callbacks=c)
    codes = Code(compressor[0], glove)
    for ep in range(epochs):
        trainer.train(loader)

        print(f"---{ep+1}th epoch---")
        for w, c in zip(words, codes.get_code(words).tolist()):
            print(f">>> {w:>8}: {c}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_component", type=int, default=8)
    p.add_argument("--num_codevec", type=int, default=8)
    p.add_argument("--sample_words", nargs="+",
                   default=["dog", "dogs", "man", "woman", "king", "queen"])
    args = p.parse_args()

    main(args.epochs, args.batch_size, args.num_component,
         args.num_codevec, args.sample_words)
