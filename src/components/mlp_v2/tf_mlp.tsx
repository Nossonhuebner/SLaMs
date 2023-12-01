import { clean } from "../../util"
import English from '../../../raw_english'
import * as tf from '@tensorflow/tfjs';

class TokenizerLite {
    map: Map<string, number>
    iToS: string[]
    constructor() {
        const chars = "*abcdefghijklmnopqrstuvwxyz.".split('')
        this.map = new Map<string, number>()
        this.iToS = chars
        for(let i = 0; i < chars.length; i++) {
            this.map.set(chars[i],  i)
        }
    }

    encode(str: string|string[]): number[] {
        // assumes string[] is already split...
        const val = typeof(str) === 'string' ? str.split('') : str
        return val.map(char => this.map.get(char) as number)
    }

    decode(vals: number[]): string {
        return vals.map(v => this.iToS[v]).join('')
    }

    get vocab() {
        return this.iToS;
    }
}

function TfMlp() {
    const tokenizer = new TokenizerLite();
    const cleaned = clean(English);
    const tokens = tokenizer.encode(cleaned);


    const inputDim = tokenizer.vocab.length+1 // Size of the vocabulary
    const embeddingSize = 2; // Size of the output embedding vector
    const contextLength = 3; // Length of each input sequence ()
    const batchSize = 64; // Number of sequences in a batch


    const { training, validation } = createDataset(tokens, contextLength)



    const model = tf.sequential();

    model.add(tf.layers.embedding({
        inputDim: inputDim,
        outputDim: embeddingSize,
        inputLength: contextLength
    }));
    model.add(tf.layers.flatten());
    tf.util.shuffle(training);
    tf.util.shuffle(validation);

    const trBatches = getBatches(training, batchSize)
    const inputs = trBatches.map(batch => batch.map(item => item.input));
    const labels = trBatches.map(batch => batch.map(item => item.output));

    const inputTensor = tf.tensor3d(inputs);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    // Predict the embeddings
    model.trainOnBatch(inputTensor, labelTensor);
    model.summary()

    return <div>Hi there</div>
}

function getBatches(dataItems: Array<DataItem>, batchSize: number) {
    const batches: Array<DataItem[]> = []
    for (let i = 0; i < dataItems.length; i += batchSize) {
        const batch = dataItems.slice(i, i + batchSize);
        batches.push(batch)
    }
    return batches;
}

function getAsEmbeddingInput(enc: number[]) {
    return tf.tensor2d(enc, [enc.length, 1]);
}

type DataItem = {
    input: number[];
    output: number;
}

function createDataset(tokens: number[], contextLength: number) {
    const training = new Array<DataItem>();
    const validation = new Array<DataItem>();

    for (let i = 0; i < tokens.length - contextLength; i++) {
        const input = tokens.slice(i, i + contextLength) as number[];
        const output = tokens[i + contextLength] as number;
        const item = { input, output };
        if (Math.random() > 0.8) {
            validation.push(item);
        } else {
            training.push(item);
        }
    }
    return { training, validation };
}

export default TfMlp;