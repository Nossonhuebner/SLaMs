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
