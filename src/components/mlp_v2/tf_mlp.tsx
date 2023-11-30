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

    const batchSize = 1;
    const embeddingSize = 2;
 

    const embeddingLayer = tf.layers.embedding({ 
        inputDim: tokenizer.vocab.length + 1, 
        outputDim: embeddingSize, 
       inputLength: batchSize
     });
}

function getAsEmbeddingInput(enc: number[]) {
    return tf.tensor2d(enc, [enc.length, 1]);
}