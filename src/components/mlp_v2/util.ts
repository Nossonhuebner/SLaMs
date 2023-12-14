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



export function predictSequences(tokenizer: ITokenizerLite, model: tf.Sequential, count: number, contextLength: number, addGenerated: (str: string) => void) { // pass in words array to allow interface to updated as each word is generated
    for (let i = 0; i < count; i++) {
        const word = predictSequence(tokenizer, model, contextLength)
        addGenerated(word)
    }
}

export function predictSequence(tokenizer: ITokenizerLite, model: tf.Sequential, contextLength: number) {
    let output = ''
    let input = "*".repeat(contextLength)
    const result = []
    while (output !== '*') {
        result.push(output)
        const outputToken = predictToken(input, tokenizer, model)
        output = tokenizer.decode([outputToken]);
        input = input.slice(1) + output
    }
    return result.join('')
}

export function predictToken(str: string, tokenizer: ITokenizerLite, model: tf.Sequential) {
    const enc = tokenizer.encode(str);
    const inputTensor = tf.tensor2d([enc], [1, str.length]);
    const prediction = model.predict(inputTensor) as tf.Tensor;  
    const values = prediction.arraySync() as number[][];
    
    const result = tf.multinomial(values, 1, undefined, true).arraySync()[0] as number;
    return result;
}