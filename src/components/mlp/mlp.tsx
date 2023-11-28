import { useMemo, useState } from "react";
import { CharacterTokenizer, ITokenizer, clean } from "../../util";
import English from '../../../raw_english'
import { Value } from "../../util/engine";
import { MLP as Net, softmax } from "../../util/nn";
import * as tf from '@tensorflow/tfjs';
import { Button, TextField, CircularProgress } from '@mui/material'

export function MLP() {
    const cleaned = clean(English);
    const tkn = new CharacterTokenizer("*abcdefghijklmnopqrstuvwxyz");
    const [lr, setLr] = useState(0.001);

    const [contextLength, setContextLength] = useState(3);
    const [embeddingSize, setEmbeddingSize] = useState(2);
    const [textLength, setTextLength] = useState(1000);
    const [running, setRunning] = useState(false)
    const embeddings = useMemo(() => createEmbeddings(tkn, embeddingSize), []);

    const net = useMemo(() => {
        return new Net(embeddingSize * contextLength, [tkn.vocabulary.length])
    }, [embeddingSize, contextLength, tkn]);

    const sample = cleaned.slice(0, textLength)

    const { training, validation } = createDataset(sample, contextLength, tkn);

    const [epochs, setEpochs] = useState(0);


    const generate = () => {
        const a = generateWords(10, net, tkn, contextLength, embeddings);
        console.log(a);    
    }

    const run = () => {
        setRunning(true)
        runEpocs(net, training, validation, embeddings, lr, epochs, () => {
            setRunning(false)
        })
    }

    return (
        <>
            lr:<TextField value={lr} onChange={e => setLr(parseFloat(e.target.value))}/>
            context:<TextField value={contextLength} onChange={e => setContextLength(parseInt(e.target.value))}/>
            emb:<TextField value={embeddingSize} onChange={e => setEmbeddingSize(parseInt(e.target.value))}/>
            epochs:<TextField value={epochs} onChange={e => setEpochs(parseInt(e.target.value))}/>
            textLength:<TextField value={textLength} onChange={e => setTextLength(parseInt(e.target.value))}/>

            <Button onClick={run}>
                {running ? <CircularProgress/> : 'run epocs'}
            </Button>
            <Button onClick={generate}>
                generate
            </Button>
        </>
    )
}

function runEpocs(net: Net, training: DataItem[], validation: DataItem[], embeddings: Value[][], lr: number, epochs: number, callBack: () => void) {
    for(let i = 0; i < epochs; i++) {
        train(net, training, embeddings, lr);
        valid(net, validation, embeddings);
    }
    callBack()
}

export function generateWords(n: number, net: Net, tkn: CharacterTokenizer, contextLength: number, embeddings: Value[][]) {
    const words = [];
    for (let i = 0; i < n; i++) {
        words.push(generateWord(net, tkn, contextLength, embeddings));
    }
    return words;
}

function generateWord(net: Net, tkn: CharacterTokenizer, contextLength: number, embeddings: Value[][]) {
    debugger
    let base = "*".repeat(contextLength);
    const chars = [];
    let next = null;
    while (next != "*") {
        const input = tkn.encode(base) as number[];
        const embs = getEmbeddings(input, embeddings);
        const logits = net.forward(embs.flat());
        const probs = softmax(logits);
        const idx = sampleProb(probs.map(v => v.data));
        next = tkn.decode([idx]);
        chars.push(next);
        base = base.slice(1) + next;
    }
    return chars.join('');
}

function sampleProb(row: number[]) {
    return tf.multinomial(row, 1, undefined, true).arraySync()[0] as number;
}


function valid(net: Net, data: DataItem[], embeddings: Value[][]) {
    const count = data.length;
    let correct = 0;
    data.forEach(item => {
        const { input, output } = item;

        const embs = getEmbeddings(input, embeddings);
        const logits = net.forward(embs.flat());
        const probs = softmax(logits);

        const yIdx = output[0];
        const probVals = probs.map(v => v.data);
        const maxProb = Math.max(...probVals)
        const predIdx = probVals.indexOf(maxProb);
        item.preds = probVals;
        correct += Number(yIdx == predIdx);
    })
    const accuracy = correct / count
    console.log('accuracy:', accuracy);
    return accuracy;
}

type DataItem = {
    input: number[],
    output: number[],
    loss?: number,
    preds?: number[]
}

function createDataset(data: string, contextLength: number, tokenizer: ITokenizer) {
    const training = new Array<DataItem>();
    const validation = new Array<DataItem>();

    for (let i = 0; i < data.length - contextLength; i++) {
        const input = tokenizer.encode(data.slice(i, i + contextLength)) as number[];
        const output = tokenizer.encode(data[i + contextLength]) as number[];
        const item = { input, output };
        if (Math.random() > 0.8) {
            validation.push(item);
        } else {
            training.push(item);
        }
    }
    return { training, validation };
}

function getEmbeddings(input: number[], embeddings: Value[][]) {
    const embedding = new Array<Value[]>();
    input.forEach(idx => {
        const e = embeddings[idx];
        embedding.push(e);
    })
    return embedding;
}

function train(net: Net, data: DataItem[], embeddings: Value[][], lr: number) {
    const count = data.length;
    let aggLoss = new Value(0)
    data.forEach(item => {
        const { input, output } = item;
        const embs = getEmbeddings(input, embeddings);
        // forward
        const logits = net.forward(embs.flat());
        const probs = softmax(logits);

        const loss = probs[output[0]].negativeLogLikelihood()
        item.loss = loss.data;
        aggLoss = aggLoss.plus(loss);

    })
    //backward
    net.parameters.forEach(p => p.grad = 0);
    aggLoss = aggLoss.divide(count);
    aggLoss.backward()

    net.parameters.forEach(p => p.data += -lr * clipGradient(p.grad, -10, 10))

    // const avg =  aggLoss / count;
    console.log('avgLoss:', aggLoss.data);
    return aggLoss.data;
}

function clipGradient(grad: number, min: number, max: number) {
    if (grad < min) return min;
    if (grad > max) return max;
    return grad;
}


function createEmbeddings(tkn: ITokenizer, embSize: number) {
    const embeddings: Value[][] = [];
    for(let i = 0; i < tkn.vocabulary.length; i++) {
        const emb: Value[] = []
        for(let j = 0; j < embSize; j++) {
            emb.push(new Value(Math.random()))
        }
        embeddings.push(emb)
    }

    return embeddings;
}

export default MLP
