import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical as GumbelSoftmax


class Encoder(nn.Module):
    def __init__(self, dim_emb, num_component, num_codevec):
        super(Encoder, self).__init__()
        self.hidden = nn.Linear(dim_emb, num_component * num_codevec // 2)
        self.out = nn.ModuleList([nn.Linear(num_component * num_codevec // 2, num_codevec)
                                  for _ in range(num_component)])
        self.register_buffer("_temperature", torch.tensor(1.0))

    def forward(self, input):
        x = F.tanh(self.hidden(input))
        x = [F.softplus(m(x)) for m in self.out]
        return [GumbelSoftmax(self._temperature, p).rsample() for p in x]

    def get_code(self, input, iteration=10):
        with torch.no_grad():
            # Gumbel softmax works in a stochastic manner, needs to be run several times to
            # get more accurate codes
            probs = sum([torch.stack(self(input)) for _ in range(iteration)])
            # num_component x batch x num_codevec
            return probs.transpose(0, 1).argmax(dim=2)

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, x):
        self._temperature.fill_(x)


class Decoder(nn.Module):
    def __init__(self, dim_emb, num_component, num_codevec):
        super(Decoder, self).__init__()
        self.emb = nn.ModuleList([nn.Linear(num_codevec, dim_emb)
                                  for _ in range(num_component)])

    def forward(self, input):
        return sum([m(v) for m, v in zip(self.emb, input)])


if __name__ == "__main__":
    dim_emb = 200
    num_component = 4
    num_codevec = 4
    a = torch.rand(3, dim_emb)
    enc = Encoder(dim_emb, num_component, num_codevec)
    dec = Decoder(dim_emb, num_component, num_codevec)
    loss = F.mse_loss(dec(enc(a)), a)
    loss.backward()
    print(enc.get_code(a))
