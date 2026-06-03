# Image Captioning on Flickr8k: A Comparative Study of Encoder–Decoder Architectures

**Piotr Salustowicz and Olgert Magilaj** — Team *Neural Narrators*
*MSE FTP — Deep Learning — Mandatory Practical Work 03*
*Both authors contributed equally to this work.*

## 1. Description of Approach

The objective of this work is to generate a meaningful natural-language description for a previously unseen image. To address this task we develop and compare two encoder–decoder captioning architectures on the Flickr8k dataset — the Show and Tell baseline of Vinyals et al. [1] and the attention-based Show, Attend and Tell model of Xu et al. [2], both built upon a single, shared, fine-tuned ResNet-18 encoder. These, together with the further decoder variants introduced later, are trained and evaluated under strictly identical conditions, so that any observed difference can be attributed to the decoder design rather than to confounding variations in the data pipeline or the training procedure. This emphasis on a fair, reproducible comparison guided every preprocessing and implementation decision described below, and reflects our view that, on a dataset of this scale, sound methodology is more informative than a marginal gain in any single metric.

A central concern of the preprocessing pipeline was the prevention of information leakage between the training and evaluation data. To this end, the caption vocabulary was constructed exclusively from the training split with a minimum word frequency of five, yielding 2,649 tokens and ensuring that no test-set vocabulary could influence the tokeniser. Each caption is delimited by reserved beginning- and end-of-sequence tokens, padded to a maximum length of forty, and mapped through four special tokens (`<pad>`, `<bos>`, `<eos>`, `<unk>`), with out-of-vocabulary words routed to the unknown token. During training all five reference captions per image are exploited as independent targets, increasing the effective training set to 32,365 image–text pairs over 6,473 unique images, whereas at evaluation a single caption is sampled from each of the 1,618 held-out images while the BLEU metric is computed against the complete reference set.

For the visual input, all images were converted to RGB and normalised using the ImageNet statistics at a resolution of 224×224. Because Flickr8k is comparatively small, the training transform applies deliberately strong augmentation to improve generalisation, comprising a random resized crop, horizontal flipping, colour jitter and occasional grayscale conversion. In contrast, the evaluation transform is fully deterministic — a resize to 256 pixels followed by a centre crop to 224 — so that reported metrics remain reproducible. Finally, to accelerate training without altering the regularisation, decoded images were cached once while augmentation continued to be applied live on each access, and the deterministic evaluation images were cached as precomputed tensors.

## 2. Architectural and Implementation Choices

In accordance with the requirement of a fair comparison, every architecture is built on the same pretrained ResNet-18 backbone, which is fine-tuned rather than frozen. For the Show and Tell baseline, the network is taken up to the adaptive average pool, projected by a linear layer and passed through batch normalisation to yield a single 512-dimensional global descriptor per image. For the Show, Attend and Tell model, the backbone is instead truncated after its final convolutional stage in order to preserve the 512×7×7 spatial feature map, which is reshaped into forty-nine spatial locations of dimension 512 over which the decoder can attend. The fine-tuned backbone accounts for roughly 11.2 million of each model's parameters (for example, 15.6 million in total for the baseline and 16.9 million for the attention model).

The two decoders share an `LSTMCell` core but differ in how they consume the visual features. The baseline decoder initialises its hidden and cell states by projecting the global image vector through two linear layers with a `tanh` non-linearity, then consumes word embeddings of dimension 256 step by step, applying dropout before a final projection to vocabulary logits. The attention decoder, in contrast, computes at every step an additive (Bahdanau) alignment over the forty-nine spatial locations; the resulting context vector is modulated by a doubly-stochastic gate and concatenated with the word embedding before entering the recurrent cell, while the initial states are derived from the mean of the spatial features. A dedicated tokeniser handles encoding and decoding, and each decoder exposes a teacher-forced `forward()` method used during training alongside a separate greedy `generate()` method used at inference, thereby satisfying the required separation between the two regimes; the attention decoder additionally returns its per-word alignment weights for later inspection.

Crucially, the pretrained backbone is optimised at a deliberately low learning rate while the randomly initialised decoder uses a higher rate, realised through separate optimiser parameter groups so that the transferred features adapt only gently to the new domain. The principal hyperparameters are summarised below.

| Hyperparameter | Baseline | Attention |
|---|---|---|
| Encoder / embedding / hidden dim | 512 / 256 / 512 | 512 / 256 / 512 |
| Attention dim | — | 256 |
| Dropout | 0.2 | 0.2 |
| Decoder LR / encoder LR | 5e-4 / 1e-5 | 5e-4 / 1e-5 |
| Weight decay | 1e-3 | 1e-3 |
| Batch size / max caption length | 64 / 40 | 64 / 40 |
| Warmup / max epochs / patience | 3 / 100 / 15 | 3 / 100 / 15 |

Beyond the two required architectures, and exceeding the minimum of two requested extensions, five additional studies were implemented under the same pipeline: beam-search decoding with length normalisation; visualisation of the attention maps overlaid on the input image for individual generated words; scaled dot-product attention as an alternative to the additive formulation; a transformer decoder with multi-head cross-attention to the spatial features, causal masking and learnable positional embeddings (four heads, three layers, trained with a peak learning rate of 3e-4 and a five-epoch warmup); and initialisation of the embedding layer from pretrained GloVe vectors, compared against a random-initialisation control under otherwise identical settings.

## 3. Experimental Setup

All models were trained with the Adam optimiser under the configuration above, using a learning-rate schedule that combines a short linear warmup with a subsequent cosine decay, together with gradient-norm clipping at 5.0 for stability. The training objective is a masked cross-entropy loss that ignores padding positions and is computed between the predicted logits and the left-shifted target tokens, so that the loss reflects only genuine caption content. To accelerate training, automatic mixed precision (bfloat16) was enabled, with all experiments conducted on a single NVIDIA H200 GPU; a complete run of one model required roughly fifteen to twenty minutes.

Regularisation was addressed through the combination of aggressive data augmentation, dropout (0.2), weight decay (1e-3) and early stopping on the validation loss with a patience of fifteen epochs. Importantly, the best-performing checkpoint is reloaded before final evaluation, ensuring that reported metrics originate from the best epoch rather than from a potentially over-fitted final epoch — a safeguard analogous in spirit to the leakage controls applied during preprocessing. Training was monitored throughout with the Weights & Biases framework, logging per-epoch loss, perplexity, token accuracy, gradient norms and the learning rate.

For evaluation we report perplexity, computed as the exponential of the token-averaged cross-entropy on the held-out set, together with corpus-level BLEU-1 to BLEU-4 obtained with NLTK and a standard smoothing function, where each generated caption is scored against all reference captions available for its image. Captions are produced by greedy decoding, with beam search evaluated separately as an extension. To preserve comparability, every variant — including the transformer decoder and the embedding study — is assessed with this identical evaluation pipeline.

## 4. Quantitative Results

All metrics below are computed on the full held-out test set using greedy decoding, after reloading each model's best validation checkpoint. Table 1 reports perplexity and corpus BLEU for the two required architectures alongside the two attention/decoder variants. Despite differing decoder designs, the four models cluster within a narrow band: perplexity ranges only from 14.21 to 14.68, and the attention-based LSTM offers essentially no improvement over the baseline on BLEU-1 (0.5883 versus 0.5871) and only a small gain at higher n-grams. The transformer decoder is the strongest model on every metric, improving BLEU-4 from 0.1782 to 0.2047 relative to the baseline, while scaled dot-product attention slightly outperforms its additive counterpart in perplexity.

**Table 1 — Perplexity and corpus BLEU on the full test set (greedy decoding).**

| Model | Best epoch | Perplexity | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|---|---|---|---|---|---|---|
| Show and Tell (baseline) | 11 | 14.61 | 0.5871 | 0.4068 | 0.2708 | 0.1782 |
| Show, Attend and Tell (Bahdanau) | 11 | 14.68 | 0.5883 | 0.4096 | 0.2774 | 0.1862 |
| Dot-Product Attention | 11 | 14.27 | 0.5915 | 0.4148 | 0.2796 | 0.1866 |
| Transformer Decoder | 9 | 14.21 | 0.6151 | 0.4340 | 0.2983 | 0.2047 |

A consistent training signature was observed across all models: the training loss and token accuracy continued to fall (the baseline reaching 57% training token accuracy), whereas the validation loss reached its minimum very early — between epochs 9 and 12 — and rose thereafter, with validation token accuracy plateauing near 41%. Early stopping therefore terminated every run well before the 100-epoch budget, and the reloaded checkpoints all originate from this early validation optimum. The corresponding training and validation curves are shown in Figure&nbsp;[X], and the four-model loss comparison in Figure&nbsp;[X].

Beam search (beam size 5, evaluated on a fixed 500-image subset for tractability) improved every BLEU order over greedy decoding on the same subset, with the largest relative gain at BLEU-4, consistent with its ability to avoid locally optimal but globally suboptimal token choices (Table 2).

**Table 2 — Greedy versus beam-search decoding (Show, Attend and Tell, 500-image subset).**

| Metric | Greedy | Beam (k = 5) |
|---|---|---|
| BLEU-1 | 0.6005 | 0.6081 |
| BLEU-2 | 0.4233 | 0.4387 |
| BLEU-3 | 0.2874 | 0.3113 |
| BLEU-4 | 0.1912 | 0.2172 |

Finally, initialising the word-embedding layer from GloVe vectors produced no practical benefit. With a vocabulary coverage of 99.5% (2,635 of 2,649 tokens matched), the GloVe-initialised model essentially matched its randomly initialised control, even falling marginally behind on BLEU (Table 3); the very high coverage indicates that Flickr8k's everyday vocabulary is learned readily from the captions themselves, leaving little headroom for transferred embeddings to contribute.

**Table 3 — Embedding initialisation (Show and Tell, embedding dim 100).**

| Metric | Random init | GloVe init |
|---|---|---|
| Perplexity | 15.43 | 15.32 |
| BLEU-1 | 0.6061 | 0.5937 |
| BLEU-2 | 0.4204 | 0.4137 |
| BLEU-3 | 0.2802 | 0.2777 |
| BLEU-4 | 0.1829 | 0.1824 |

## 5. Qualitative Analysis

To complement the aggregate metrics, greedy captions were generated for held-out images from every model, attention overlays were produced for the Show, Attend and Tell model on representative test images, and beam-search and greedy outputs were inspected side by side. The shape of the BLEU profile is itself informative: across all models BLEU-1 is moderate (≈0.59–0.62) while BLEU-4 is substantially lower (≈0.18–0.20), a steep decay indicating that the models reliably identify and name the salient objects of a scene but are markedly weaker at producing the longer, fluent and correctly ordered phrasing that higher-order n-grams reward. This is the behaviour expected of a small-data captioner and frames the error analysis below.

We assess the generated captions against the error categories specified in the assignment — missing objects, hallucinated objects, repetitive text, incorrect relationships and overly generic captions. Representative successful and failure cases are presented in Figure&nbsp;[X] (sample captions per model) and Figure&nbsp;[X] (attention overlays).

`[PLACEHOLDER: Insert 3–4 representative generated captions (one or two successes, one or two failures) with their images, and classify each failure by error type. State, from the attention figures, whether the Show, Attend and Tell model attends to semantically relevant regions (e.g. the object being named) or diffuses its attention, and note any words for which attention was clearly interpretable. This subjective reading must be written from your own inspection of the figures.]`

## 6. Discussion of Limitations and Possible Improvements

The most striking finding is methodological rather than architectural: every model — regardless of decoder design — reached its validation optimum within the first dozen epochs and then over-fitted, despite the combined use of strong augmentation, dropout, weight decay and early stopping. The persistent gap between training token accuracy (rising past 55%) and validation token accuracy (plateauing near 41%) confirms that the binding constraint is the size of the dataset, not the capacity of the models. This directly explains the narrow spread of results and is consistent with the assignment's expectation that only limited performance is achievable on a dataset of this scale.

Within this regime, the comparison between architectures is nonetheless informative. Spatial attention provided only a marginal improvement over the global-feature baseline, suggesting that with a fine-tuned encoder the single pooled descriptor already captures most of the information the decoder can exploit before over-fitting intervenes; the choice between additive and scaled dot-product attention was likewise near-immaterial. The transformer decoder was the clearest, if still modest, improvement, which we attribute to its more expressive token-mixing rather than to any change in the visual representation, since it reuses the same encoder. Beam search was the most reliable lever on caption quality, improving every BLEU order and most strongly the higher-order scores that reflect fluency.

Several limitations temper these conclusions. BLEU rewards n-gram overlap and is an imperfect proxy for caption adequacy, so the small metric differences observed should not be over-interpreted; metrics such as CIDEr or METEOR, or a human preference study, would characterise quality more faithfully. The beam-search and embedding comparisons were run on a subset and at a reduced embedding dimension respectively, and greedy decoding remains the inference strategy for the headline comparison. Promising directions for improvement follow directly from the leakage- and overfitting-aware framing of this work: enlarging or supplementing the dataset (for instance with a larger captioning corpus or stronger augmentation), exploring a more powerful or larger-resolution encoder, scheduled sampling to reduce exposure bias between teacher-forced training and free-running inference, and label smoothing or stronger regularisation tuned specifically to the early-overfitting behaviour identified above.

## References

[1] O. Vinyals, A. Toshev, S. Bengio, D. Erhan. *Show and Tell: A Neural Image Caption Generator.* 2015. arXiv:1411.4555.
[2] K. Xu et al. *Show, Attend and Tell: Neural Image Caption Generation with Visual Attention.* ICML, 2015. arXiv:1502.03044.
[3] D. Bahdanau, K. Cho, Y. Bengio. *Neural Machine Translation by Jointly Learning to Align and Translate.* ICLR, 2015. arXiv:1409.0473.
[4] A. Vaswani et al. *Attention Is All You Need.* NeurIPS, 2017. arXiv:1706.03762.
[5] J. Pennington, R. Socher, C. Manning. *GloVe: Global Vectors for Word Representation.* EMNLP, 2014.
[6] K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition.* CVPR, 2016. arXiv:1512.03385.
