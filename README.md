# neuralcompressor.pytorch

An implementation of neuralcompressor, proposed in [*Compressing Word Embeddings via Deep Compositional Code Learning*](http://arxiv.org/abs/1711.01068) by Raphael Shu and Hideki Nakayama (ICLR, 2018).

The status of this repository is *WORK IN PROGRESS*.

# Prerequirements

* `Python >= 3.6`
* `PyTorch >= 4.0`
* `torchtext` 

```
git clone https://github.com/pytorch/text.git
cd text
pip install -e .
```

# Results
## Qualitative Results

`glove.py` compresses GloVe-6b 300 dim word vectors into codings.

```
python glove.py --epochs 10 --batch_size 128 --num_component 8 --num_codevec 8 --sample_words dog dogs man woman king queen
```


```
  dog: [3, 1, 2, 5, 3, 0, 0, 7]
 dogs: [3, 1, 1, 1, 3, 0, 0, 3]
  man: [3, 1, 2, 2, 3, 6, 0, 0]
woman: [3, 1, 2, 1, 3, 2, 0, 4]
 king: [2, 1, 6, 3, 3, 4, 0, 0]
queen: [2, 1, 4, 3, 3, 2, 0, 4]
```
