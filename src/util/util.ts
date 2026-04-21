import * as tf from '@tensorflow/tfjs';
import { ITokenizerLite } from './tokenizers';

export type DataItem = {
    input: number[];
    output: number;
}


export function createDataset(tokens: number[], contextLength: number) {
    const r = new Array<DataItem>();

    for (let i = 0; i < tokens.length - contextLength; i++) {
        const input = tokens.slice(i, i + contextLength) as number[];
        const output = tokens[i + contextLength] as number;
        const item = { input, output };
        r.push(item);
    }
    return r;
}



export function predictSequences(tokenizer: ITokenizerLite, model: tf.LayersModel, count: number, contextLength: number, addGenerated: (str: string) => void) {
    for (let i = 0; i < count; i++) {
        const word = predictSequence(tokenizer, model, contextLength)
        addGenerated(word)
    }
}

export function predictSequence(tokenizer: ITokenizerLite, model: tf.LayersModel, contextLength: number, startText?: string) {
    let output: number | null = null

    // Initialize with start text or spaces
    let input = startText ?
        tokenizer.encode(startText) :
        tokenizer.encode(" ".repeat(contextLength))

    // Validate input length
    if (input.length !== contextLength) {
        input = input.slice(-contextLength).concat(
            Array(Math.max(0, contextLength - input.length)).fill(
                tokenizer.encode(" ")[0]
            )
        )
    }

    const result: number[] = []

    let safety = 0
    const MAX_LENGTH = 30 // Maximum sequence length
    const TEMPERATURE = 0.8 // Slightly increased temperature for more variety

    while (safety < MAX_LENGTH) {
        // Validate input length before prediction
        if (input.length !== contextLength) {
            console.error("Invalid input length during generation:", input.length)
            break
        }

        // Get next token prediction
        const nextToken = predictToken(input, model, TEMPERATURE)[0];
        if (nextToken === undefined) {
            console.error("Failed to predict next token");
            break;
        }

        output = nextToken;

        // Stop if we hit the special token
        if (output === tokenizer.specialToken) {
            break;
        }

        input = [...input.slice(1), output]
        result.push(output)
        safety++
    }

    if (result.length === 0) {
        return startText || "";
    }

    return tokenizer.decode(result)
}

export function predictToken(inputTokens: number[], model: tf.LayersModel, temperature: number = 1.0) {
    return tf.tidy(() => {
        // For embedding layer, input should be int32
        const inputTensor = tf.tensor2d([inputTokens], [1, inputTokens.length], 'int32');

        const prediction = model.predict(inputTensor) as tf.Tensor;
        const values = prediction.arraySync() as number[][];

        // Ensure we have valid probabilities
        if (!values || !values[0] || values[0].some(isNaN)) {
            console.error("Invalid prediction values:", values);
            return [0]; // Return space token index as fallback
        }

        // Apply temperature scaling
        const logits = values[0].map(x => Math.log(Math.max(x, 1e-8)) / temperature);

        // Ensure valid probability distribution
        const maxLogit = Math.max(...logits);
        const expLogits = logits.map(x => Math.exp(x - maxLogit));
        const sum = expLogits.reduce((a, b) => a + b, 0);
        const probs = expLogits.map(x => x / sum);

        // Create tensor and sample
        const probTensor = tf.tensor1d(probs);
        const result = tf.multinomial(probTensor, 1).arraySync() as number[];

        return result;
    });
}

// Model hyperparameters
const MODEL_CONFIG = {
    EMBEDDING_SIZE: 32,
    DROPOUT_RATE: 0.1,
    L2_REGULARIZATION: 0.0001
}

export function createModel(inputSize: number, outputSize: number, hiddenSize: number) {
    const model = tf.sequential();

    // Embedding layer
    model.add(tf.layers.embedding({
        inputDim: outputSize,
        outputDim: MODEL_CONFIG.EMBEDDING_SIZE,
        inputLength: inputSize
    }));

    // Flatten the embedding output
    model.add(tf.layers.flatten());

    // First hidden layer
    model.add(tf.layers.dense({
        units: hiddenSize,
        activation: 'relu',
        kernelInitializer: 'glorotUniform',
        kernelRegularizer: tf.regularizers.l2({ l2: MODEL_CONFIG.L2_REGULARIZATION })
    }));
    model.add(tf.layers.dropout({ rate: MODEL_CONFIG.DROPOUT_RATE }));

    // Second hidden layer
    model.add(tf.layers.dense({
        units: hiddenSize / 2,
        activation: 'relu',
        kernelInitializer: 'glorotUniform',
        kernelRegularizer: tf.regularizers.l2({ l2: MODEL_CONFIG.L2_REGULARIZATION })
    }));
    model.add(tf.layers.dropout({ rate: MODEL_CONFIG.DROPOUT_RATE }));

    // Output layer
    model.add(tf.layers.dense({
        units: outputSize,
        activation: 'softmax',
        kernelInitializer: 'glorotUniform'
    }));

    return model;
}
