# SLaMs
(Smol Language Models)

## Overview
This repository is a collection of small language models that train directly in the browser, showcasing the progression from simple to advanced NLP techniques. Each model is implemented in a React component, demonstrating both linguistic theory and software engineering practices.

### Projects Included:
- Bigram Language Model: Starting with the basics, this model demonstrates the fundamental concepts of language modeling using bigrams.
- Custom MLP Language Model: A handcrafted TypeScript implementation of a Multi-Layer Perceptron (MLP) combined with embeddings. This basic language model lays the groundwork for more complex architectures.
- MLP with TensorFlow.js: Enhancing our previous MLP model, this implementation leverages TensorFlow.js for improved handling of modeling, batching, and embeddings - especially since the handrolled version does not train very well :)
- WaveNet (Under Construction): A project to implement a WaveNet model
- Transformer (Coming Soon): Implementation of a transformer model, set to push the boundaries of our in-browser NLP capabilities.

#### TODO:
   - [ ] For each model / component, add a word-tokenizer for next word predictions, in addition to the current character-level tokens
   - [ ] Bigram - add support for larger n-grams
   - [ ] Improve training for handrolled basic MLP model
   - [ ] Complete WaveNet model
   - [ ] Complete Transformer model
   - [ ] Add character / word tokenizers for the original Hebrew version of the dataset

The dataset is the translated Bible text taken from https://mechon-mamre.org/, (used `document.querySelectorAll('.h').forEach(e => e.remove());` to remove the hebrew prior to copying).

## Appendix: LLM-Assisted Follow-ups & Debugging

A log of changes driven by pair-programming sessions with Claude. Each entry links to the commit and summarizes what was addressed.

- [`2e268a4`](https://github.com/Nossonhuebner/SLaMs/commit/2e268a44879cb8bf06c36c33b7d682ad6e6d5861) — Review & repair of in-flight changes across the MLP and RNN components: restored the TF-based `/mlp-v2` route and added `/mlp-v3`; replaced the TF MLP's broken `JSON.parse(model)` worker contract with an architecture-params + weight-arrays protocol so training actually runs off the main thread; dropped a learning-rate schedule in `tf_2` that was resetting Adam state every epoch (and running LR=0 at epoch 0); fixed the RNN by adding an embedding layer and unifying train/generate shapes; widened model types in `util.ts`. Also introduced a Vitest test suite (`tests/`) covering the three tokenizer implementations, `createDataset` in both shapes, `clean()` (English + Hebrew paths), and the Genesis corpus invariants.
- [`3fb8f03`](https://github.com/Nossonhuebner/SLaMs/commit/3fb8f030c595176af414c1f358a99b4438206eac) — Unblocked the GitHub Pages deploy after `actions/upload-artifact@v3` was disabled: bumped the full Pages action chain (`configure-pages` → v5, `upload-pages-artifact` → v3, `deploy-pages` → v4) plus `checkout`/`setup-node` to v4, and moved Node from EOL 18 to 20.
