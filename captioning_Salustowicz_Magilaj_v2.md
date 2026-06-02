# Image Captioning on Flickr8k: A Comparative Study of Encoder–Decoder Architectures

**Piotr Salustowicz and Olgert Magilaj**
*MSE FTP — Deep Learning — Mandatory Practical Work 03*
*Both authors contributed equally to this work.*

## Abstract

Automatically generating a natural-language description of an image requires a model to jointly interpret visual content and produce coherent text, a task that remains challenging on small datasets. In this work we develop and compare two encoder–decoder captioning architectures on the Flickr8k dataset: the Show and Tell baseline of Vinyals et al. and the attention-based Show, Attend and Tell model of Xu et al., both built upon a single, shared, fine-tuned ResNet-18 encoder. To isolate the effect of the decoder design, the two systems share an identical, leakage-free data pipeline together with a common training, decoding and evaluation protocol, and the comparison is extended with beam search, an alternative attention mechanism, a transformer decoder and pretrained word embeddings. Given the limited size of the dataset, this study deliberately emphasises methodological rigour and transparent evaluation over the pursuit of maximal metric scores.

## 1. Description of Approach

The objective of this work is to generate a meaningful natural-language description for a previously unseen image. To address this task in a controlled manner, we adopt a comparative experimental design in which two encoder–decoder architectures are trained and evaluated under strictly identical conditions, so that any observed difference can be attributed to the decoder design rather than to confounding variations in the data pipeline or the training procedure. This emphasis on a fair, reproducible comparison guided every preprocessing and implementation decision described below, and reflects our view that, on a dataset of this scale, sound methodology is more informative than a marginal gain in any single metric.

A central concern of the preprocessing pipeline was the prevention of information leakage between the training and evaluation data. To this end, the caption vocabulary was constructed exclusively from the training split with a minimum word frequency of five, ensuring that no test-set vocabulary could influence the tokeniser. Each caption is delimited by reserved beginning- and end-of-sequence tokens, padded to a maximum length of forty, and mapped through four special tokens (`<pad>`, `<bos>`, `<eos>`, `<unk>`), with out-of-vocabulary words routed to the unknown token. During training all five reference captions per image are exploited as independent targets, thereby increasing the effective number of image–text pairs, whereas at evaluation a single caption is sampled per image while the BLEU metric is computed against the complete reference set.

For the visual input, all images were converted to RGB and normalised using the ImageNet statistics at a resolution of 224×224. Because Flickr8k is comparatively small, the training transform applies deliberately strong augmentation to improve generalisation, comprising a random resized crop, horizontal flipping, colour jitter and occasional grayscale conversion. In contrast, the evaluation transform is fully deterministic — a resize to 256 pixels followed by a centre crop to 224 — so that reported metrics remain reproducible. Finally, to accelerate training without altering the regularisation, decoded images were cached once while augmentation continued to be applied live on each access, and the deterministic evaluation images were cached as precomputed tensors.

## 2. Architectural and Implementation Choices

In accordance with the requirement of a fair comparison, both architectures are built on the same pretrained ResNet-18 backbone, which is fine-tuned rather than frozen. For the Show and Tell baseline, the network is taken up to the adaptive average pool, projected by a linear layer and passed through batch normalisation to yield a single 512-dimensional global descriptor per image. For the Show, Attend and Tell model, the backbone is instead truncated after its final convolutional stage in order to preserve the 512×7×7 spatial feature map, which is reshaped into forty-nine spatial locations of dimension 512 over which the decoder can attend.

The two decoders share an `LSTMCell` core but differ in how they consume the visual features. The baseline decoder initialises its hidden and cell states by projecting the global image vector through two linear layers with a `tanh` non-linearity, then consumes word embeddings of dimension 256 step by step, applying dropout before a final projection to vocabulary logits. The attention decoder, in contrast, computes at every step an additive (Bahdanau) alignment over the forty-nine spatial locations; the resulting context vector is modulated by a doubly-stochastic gate and concatenated with the word embedding before entering the recurrent cell, while the initial states are derived from the mean of the spatial features. A dedicated tokeniser handles encoding and decoding, and each decoder exposes a teacher-forced `forward()` method used during training alongside a separate greedy `generate()` method used at inference, thereby satisfying the required separation between the two regimes; the attention decoder additionally returns its per-word alignment weights for later inspection.

Crucially, the pretrained backbone is optimised at a deliberately low learning rate while the randomly initialised decoder uses a higher rate, realised through separate optimiser parameter groups so that the transferred features adapt only gently to the new domain. The principal hyperparameters are summarised below.

| Hyperparameter | Baseline | Attention |
|---|---|---|
| Encoder / embedding / hidden dim | 512 / 256 / 512 | 512 / 256 / 512 |
| Attention dim | — | 256 |
| Dropout | 0.1 | 0.1 |
| Decoder LR / encoder LR | 5e-4 / 1e-5 | 5e-4 / 1e-5 |
| Weight decay | 1e-5 | 1e-5 |
| Batch size / max caption length | 100 / 40 | 100 / 40 |
| Warmup / max epochs / patience | 3 / 100 / 8 | 3 / 100 / 8 |

Beyond the two required architectures, and exceeding the minimum of two requested extensions, five additional studies were implemented under the same pipeline: beam-search decoding with length normalisation; visualisation of the attention maps overlaid on the input image for individual generated words; scaled dot-product attention as an alternative to the additive formulation; a transformer decoder with multi-head cross-attention to the spatial features, causal masking and learnable positional embeddings; and initialisation of the embedding layer from pretrained GloVe vectors, compared against a random-initialisation control under otherwise identical settings.

## 3. Experimental Setup

All models were trained with the Adam optimiser under the configuration above, using a learning-rate schedule that combines a short linear warmup with a subsequent cosine decay, together with gradient-norm clipping for stability. The training objective is a masked cross-entropy loss that ignores padding positions and is computed between the predicted logits and the left-shifted target tokens, so that the loss reflects only genuine caption content. To accelerate training, automatic mixed precision was enabled, with the experiments conducted on a single GPU.

Regularisation was addressed through the combination of aggressive data augmentation, dropout, weight decay and early stopping on the validation loss. Importantly, the best-performing checkpoint is reloaded before final evaluation, ensuring that reported metrics originate from the best epoch rather than from a potentially over-fitted final epoch — a safeguard analogous in spirit to the leakage controls applied during preprocessing. Training was monitored throughout with the Weights & Biases framework, logging per-epoch loss, perplexity, token accuracy, gradient norms and the learning rate.

For evaluation we report perplexity, computed as the exponential of the token-averaged cross-entropy on the held-out set, together with corpus-level BLEU-1 to BLEU-4 obtained with NLTK and a standard smoothing function, where each generated caption is scored against all reference captions available for its image. Captions are produced by greedy decoding, with beam search evaluated separately as an extension. To preserve comparability, every variant — including the transformer decoder and the embedding study — is assessed with this identical evaluation pipeline.

## 4. Quantitative Results

`[PLACEHOLDER: Quantitative Results — Insert generated metric scores here once execution completes]`

*(Intended contents: a table of perplexity and BLEU-1…4 for the baseline, the attention model and the extension variants; training/validation loss and validation-perplexity curves over epochs; and the greedy-versus-beam-search BLEU comparison.)*

## 5. Qualitative Analysis

`[PLACEHOLDER: Qualitative Analysis — Insert sample generated captions and evaluation of their quality here]`

*(Intended contents: representative successful and failure cases for each model; attention-heatmap overlays for selected words; and a discussion of typical error modes — missing objects, hallucinated objects, repetition, incorrect relationships and overly generic captions.)*

## 6. Discussion of Limitations and Possible Improvements

`[PLACEHOLDER: Discussion of Limitations and Possible Improvements — Detail methodology constraints, model limits, and acknowledge the limited dataset performance here]`

*(Intended contents: constraints imposed by the small Flickr8k dataset, the limitations of greedy decoding and of BLEU as a captioning metric, observations on encoder fine-tuning, and concrete directions for improvement.)*
