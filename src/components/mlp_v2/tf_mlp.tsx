import { clean } from "../../util"
import English from '../../../raw_english'
import * as tf from '@tensorflow/tfjs';
import { useMemo, useState } from "react";
import { Button } from "@mui/material";                 

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

const embeddingSize = 30; // Size of the output embedding vector
const contextLength = 5; // Length of each input sequence ()
const batchSize = 640; // Number of sequences in a batch

const lr = 0.001

function TfMlp() {
    const tokenizer = new TokenizerLite();
    const cleaned = clean(English);
    console.log(cleaned.length)
    const sample = cleaned.slice(0, 30000);
    const tokens = tokenizer.encode(sample);
    const [generated, setGenerated] = useState<string[]>([]);

    const inputDim = tokenizer.vocab.length+1 // Size of the vocabulary


    const d = createDataset2(tokens, contextLength)
    tf.util.shuffle(d);
    const inn = d.map(item => item.input);
    const out = d.map(item => item.output);

    const model = useMemo(() => {
        const m = tf.sequential();

        // Add an embedding layer
        m.add(tf.layers.embedding({
            inputDim: inputDim, // vocabulary size
            outputDim: embeddingSize, // embedding size
            inputLength: contextLength // context size
        }));

        // Flatten the output from the embedding layer
        m.add(tf.layers.flatten());
        m.add(tf.layers.dense({ units: 64, activation: 'relu' }));
        m.add(tf.layers.dense({ units: 28, activation: 'softmax' }));


        m.compile({
            optimizer:  tf.train.adam(lr) ,
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });

        return m;
    }, [inputDim, embeddingSize, contextLength]);



    model.summary()
    const oneHotLabels = tf.oneHot(out, tokenizer.vocab.length);

    const train = () => {
        model.fit(tf.tensor2d(inn, [inn.length, contextLength]), oneHotLabels, {
            epochs: 30,        // Number of epochs to train the model
            validationSplit: 0.2,  // Fraction of the training data to be used as validation data
            batchSize: batchSize,
            callbacks: {       // Callbacks to execute during training
              onEpochEnd: (epoch, logs) => {
                console.log(logs)
                console.log(`Epoch ${epoch + 1}: Loss: ${logs?.loss}, Val_Accuracy: ${logs?.val_acc}`);
              }
            }
        });
    
    }


    return (
        <>
            <div>Hi there</div>
            <Button onClick={train}>train</Button>
            <Button onClick={() => predictWords(tokenizer ,model, 10, generated, setGenerated)}>generate</Button>
            {generated.map((g, i) => <div key={i}>{g}</div>)}
        </>
    )
}

function predictWords(tokenizer: TokenizerLite, model: tf.Sequential, count: number, words: string[], setWords: React.Dispatch<React.SetStateAction<string[]>>) { // pass in words array to allow interface to updated as each word is generated
    for (let i = 0; i < count; i++) {
        setWords([...words, predictWord(tokenizer, model)])
    }
}

function predictWord(tokenizer: TokenizerLite, model: tf.Sequential) {
    let output = ''
    let input = "*".repeat(contextLength)
    const result = []
    while (output !== '*') {
        result.push(output)
        output = predict(input, tokenizer, model)
        input = input.slice(1) + output
    }
    return result.join('')
}


function predict(str: string, tokenizer: TokenizerLite, model: tf.Sequential) {
    const enc = tokenizer.encode(str);
    const inputTensor = tf.tensor2d([enc], [1, contextLength]);
    const prediction = model.predict(inputTensor) as tf.Tensor;  
    const values = prediction.arraySync() as number[][];
    
    const result = tf.multinomial(values, 1, undefined, true).arraySync()[0] as number;
    return tokenizer.decode([result]);
}

type DataItem = {
    input: number[];
    output: number;
}


function createDataset2(tokens: number[], contextLength: number) {
    const r = new Array<DataItem>();

    for (let i = 0; i < tokens.length - contextLength; i++) {
        const input = tokens.slice(i, i + contextLength) as number[];
        const output = tokens[i + contextLength] as number;
        const item = { input, output };
        r.push(item);
    }
    return r;
}

export default TfMlp;