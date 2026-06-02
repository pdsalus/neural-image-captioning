# Image Captioning on Flickr8k: A Comparative Study of Encoder–Decoder Architectures

**Authors:** Piotr Salustowicz & Olgert Magilaj
**MSE FTP — Deep Learning — Mandatory Practical Work 03**

---

## 1. Description of Approach

We develop and compare two image-captioning architectures on the Flickr8k dataset (~6k training images, five human-written captions each), using the fixed train/test split provided with the assignment. Our overarching strategy is methodological rather than performance-driven: both models share an identical data pipeline, the same pretrained image encoder, and the same training, decoding and evaluation procedures, so that any measured difference is attributable to the decoder design rather than to confounding setup choices. Given the small dataset, we prioritise a controlled, leakage-free experimental design and honest reporting over maximising metric scores.

**Caption preprocessing.** The vocabulary is built **exclusively from the training split** (`min_freq = 5`), which prevents test-set leakage into the tokeniser. Four reserved tokens structure every sequence: `<pad>` (0), `<bos>` (1), `<eos>` (2) and `<unk>` (3); out-of-vocabulary words map to `<unk>`. Captions are wrapped with `<bos>/<eos>` and padded to a maximum length of 40. During training all five reference captions per image are used (`caption_sampling="all"`), increasing the effective number of image–text pairs; at test time a single caption is sampled per image while BLEU is computed against the full reference set.

**Image preprocessing and augmentation.** All images are converted to RGB and normalised with ImageNet mean/std at 224×224. Because the dataset is small, the training transform applies aggressive augmentation for regularisation: `RandomResizedCrop(224, scale=(0.65, 1.0), ratio=(0.75, 1.33))`, horizontal flip (p = 0.5), colour jitter (brightness/contrast/saturation = 0.3, hue = 0.1) and occasional grayscale (p = 0.05). The test transform is deterministic — `Resize(256) → CenterCrop(224)` with no augmentation — to keep evaluation reproducible. To accelerate training without altering the regularisation, decoded and resized images are cached once while augmentation is still applied live per access; the deterministic test images are cached as precomputed tensors.

## 2. Architectural and Implementation Choices

**Shared encoder.** Both architectures use the **same pretrained ResNet-18** backbone (ImageNet weights), which is *fine-tuned* rather than frozen. For the baseline, the network is taken up to the adaptive average pool, flattened, projected with a linear layer to a 512-d vector and passed through `BatchNorm1d`, yielding one global descriptor per image. For the attention model, the backbone is truncated after `layer4` to preserve the 512×7×7 convolutional feature map, reshaped to 49 spatial locations of dimension 512. Keeping a single encoder design ensures a fair comparison, as required by the assignment.

**Model 1 — Show and Tell (baseline).** An `LSTMCell` decoder (hidden size 512) is initialised by projecting the global image vector into the initial hidden and cell states via two linear layers with `tanh`. Word embeddings (dimension 256) are fed step-by-step; dropout is applied to embeddings and to the hidden state before a linear projection to vocabulary logits.

**Model 2 — Show, Attend and Tell.** At each decoding step, an **additive (Bahdanau) attention** module computes a soft alignment over the 49 spatial locations, `energy = vᵀ tanh(W_enc·a + W_dec·h)`, normalised by softmax to produce a context vector. A **doubly-stochastic gate** (`σ(W·h)`) modulates the context, which is concatenated with the word embedding before entering the `LSTMCell` (input size 256 + 512). Hidden/cell states are initialised from the mean of the spatial features.

**Tokenisation, teacher forcing and inference.** A dedicated `CaptionTokenizer` handles encoding/decoding and special-token bookkeeping. Each decoder exposes a `forward()` method (teacher-forced, returning logits of shape `B × (T−1) × V`) and a separate `generate()` method (greedy decoding, selecting the arg-max token at every step), satisfying the required train/inference separation. The attention decoder additionally provides `generate_with_attention()` for inspecting per-word alignment weights.

**Fine-tuning configuration.** The pretrained backbone is unfrozen and optimised with a deliberately low learning rate (1e-5), while the randomly initialised decoder uses a higher rate (5e-4); these are realised as separate Adam parameter groups so the pretrained features adapt gently. Key hyperparameters are summarised below.

| Hyperparameter | Baseline | Attention |
|---|---|---|
| Encoder / embedding / hidden dim | 512 / 256 / 512 | 512 / 256 / 512 |
| Attention dim | — | 256 |
| Dropout | 0.1 | 0.1 |
| Decoder LR / Encoder LR | 5e-4 / 1e-5 | 5e-4 / 1e-5 |
| Weight decay | 1e-5 | 1e-5 |
| Batch size / max caption len | 100 / 40 | 100 / 40 |
| Warmup epochs / max epochs / patience | 3 / 100 / 8 | 3 / 100 / 8 |

**Extensions.** Beyond the two required architectures we implement five extensions (the assignment requires at least two): beam-search decoding (beam size 5, length-normalised); attention-map visualisation (7×7 maps bilinearly upsampled and overlaid per generated word); scaled dot-product attention as an alternative to Bahdanau attention; a transformer decoder with multi-head cross-attention to the spatial features (4 heads, 3 layers, FFN 1024, pre-norm, causal masking, learnable positional embeddings); and GloVe-6B-100d embedding initialisation compared against a random-init control under identical hyperparameters.

## 3. Experimental Setup

All models are trained with Adam under the configuration above, using a learning-rate schedule of linear warmup followed by cosine decay to 10 % of the peak rate, and gradient-norm clipping at 5.0 for stability. The loss is **masked cross-entropy** that ignores `<pad>` positions, computed between the logits and the left-shifted target tokens (`captions[:, 1:]`). Training uses automatic mixed precision (bf16 on supported GPUs, otherwise fp16 with a gradient scaler) on a single NVIDIA L40S GPU. Regularisation combines data augmentation, dropout, weight decay and **early stopping** (patience 8 on validation loss); the **best checkpoint is reloaded before final evaluation**, so reported metrics reflect the best epoch rather than a potentially over-fitted final epoch. Training is monitored with Weights & Biases (per-epoch loss, perplexity, token accuracy, gradient norms and learning rate).

For evaluation we report **perplexity** — the exponential of the token-averaged cross-entropy on the held-out set — and **corpus-level BLEU-1 to BLEU-4** (NLTK `corpus_bleu` with smoothing method 1), where each generated caption is scored against all reference captions for its image. Captions are produced by greedy decoding (max length 35), with beam search evaluated separately as an extension. All four LSTM-based and transformer variants are evaluated with the identical pipeline to keep results directly comparable.

## 4. Quantitative Results

`[PLACEHOLDER: Quantitative Results — Insert generated metric scores here once execution completes]`

*(Intended contents: a table of perplexity and BLEU-1…4 for the baseline, the attention model, and the extension variants; training/validation loss curves and validation-perplexity curves over epochs; and the greedy-vs-beam-search BLEU comparison.)*

## 5. Qualitative Analysis

`[PLACEHOLDER: Qualitative Analysis — Insert sample generated captions and evaluation of their quality here]`

*(Intended contents: representative successful and failure cases for each model; attention heatmap overlays for selected words; and a discussion of typical error modes — missing objects, hallucinated objects, repetition, incorrect relationships and overly generic captions.)*

## 6. Discussion of Limitations and Possible Improvements

`[PLACEHOLDER: Discussion of Limitations and Possible Improvements — Detail methodology constraints, model limits, and acknowledge the limited dataset performance here]`

*(Intended contents: constraints imposed by the small Flickr8k dataset, limitations of greedy decoding and of BLEU as a captioning metric, observations on encoder fine-tuning, and concrete directions for improvement.)*
