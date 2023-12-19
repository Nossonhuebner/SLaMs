import { useMemo, useState } from "react";
import { clean } from "../../util";
import English from '../../../raw_english'
import { Value } from "../../util/engine";
import { MLP as Net, softmax } from "../../util/nn";
import * as tf from '@tensorflow/tfjs';
import { Button, Grid, TextField, CircularProgress } from '@mui/material'
import { CharTokenizerLite, ITokenizerLite } from "../../util/tokenizers";
import { DataItem, createDataset } from "../../util/util";

import './styles.css'

export function MLP() {
    window.tf = tf;


    const [lr, setLr] = useState(0.001);
    const [batchSize, setBatchSize] = useState(500);
    const [contextLength, setContextLength] = useState(3);  // Length of each input sequence
    const [embeddingSize, setEmbeddingSize] = useState(10);
    const [sampleSize, setSampleSize] = useState(2000);
    const [epochs, setEpochs] = useState(5);

    const tokenizer = useMemo(() => new CharTokenizerLite(), []);
    const cleaned = clean(English);
    const sample = cleaned.slice(0, sampleSize);
    const tokens = tokenizer.encode(sample);
    // const [generated, setGenerated] = useState<string[]>([]);

    const dataset = createDataset(tokens, contextLength)
    const vocabSize = tokenizer.vocab.length;
    const [running, setRunning] = useState(false)
    const embeddings = useMemo(() => createEmbeddings(tokenizer, embeddingSize), [embeddingSize]);

    const net = useMemo(() => {

        return new Net(embeddingSize * contextLength, [50, vocabSize])
    }, [embeddingSize, contextLength]);


    const generate = () => {
        const a = generateWords(10, net, tokenizer, contextLength, embeddings);
        console.log(a);    
    }

    const run = () => {
        setRunning(true)
        runEpocs(net, dataset, embeddings, lr, epochs, batchSize, () => {
            setRunning(false)
        })
    }

    return (
        <>
            <Grid container className="inputContainer">
                <Grid item xs={6}><TextField label={`Sample Size (max ${cleaned.length.toLocaleString('en-US')})`} value={sampleSize} onChange={e => setSampleSize(parseInt(e.target.value))} /></Grid>
                <Grid item xs={6}><TextField label="Batch Size" value={batchSize} onChange={e => setBatchSize(parseInt(e.target.value))} /></Grid>
                

                <Grid item xs={6}><TextField label="Context Length" value={contextLength} onChange={e => setContextLength(parseInt(e.target.value))} /></Grid>
                <Grid item xs={6}><TextField label="Embedding Size" value={embeddingSize} onChange={e => setEmbeddingSize(parseInt(e.target.value))} /></Grid>
                
                <Grid item xs={6}><TextField label="Learning Rate" value={lr} onChange={e => setLr(parseFloat(e.target.value))} /></Grid>
                <Grid item xs={6}><TextField label="Epochs" value={epochs} onChange={e => setEpochs(parseInt(e.target.value))} /></Grid>
            </Grid>

            <Button onClick={run}>
                {running ? <CircularProgress/> : 'run epocs'}
            </Button>
            <Button onClick={generate}>
                generate
            </Button>
        </>
    )
}

function runEpocs(net: Net, dataset: DataItem[], embeddings: Value[][], lr: number, epochs: number, batchSize: number, callBack: () => void) {
    for(let i = 0; i < epochs; i++) {
        const [training, validation] = split(shuffle(dataset));
        console.log('epoch:', i);
        batch(training, batchSize).forEach(batch => {
            train(net, batch, embeddings, lr);
        })
        batch(validation, batchSize).forEach(batch => {
            valid(net, batch, embeddings);
        })
    }
    callBack()
}

export function generateWords(n: number, net: Net, tkn: ITokenizerLite, contextLength: number, embeddings: Value[][]) {
    const words = [];
    for (let i = 0; i < n; i++) {
        words.push(generateWord(net, tkn, contextLength, embeddings));
    }
    return words;
}

function generateWord(net: Net, tkn: ITokenizerLite, contextLength: number, embeddings: Value[][]) {
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

        const yIdx = output;
        const probVals = probs.map(v => v.data);
        const maxProb = Math.max(...probVals)
        const predIdx = probVals.indexOf(maxProb);
        correct += Number(yIdx == predIdx);
    })
    const accuracy = correct / count
    console.log('accuracy:', accuracy);
    return accuracy;
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

        const loss = probs[output].negativeLogLikelihood()
        aggLoss = aggLoss.plus(loss);
    })
    //backward
    net.parameters.forEach(p => p.grad = 0);
    aggLoss = aggLoss.divide(count);
    aggLoss.backward()

    net.parameters.forEach(p => p.data += -lr * clipGradient(p.grad, -5, 5))

    // const avg =  aggLoss / count;
    console.log('avgLoss:', aggLoss.data);
    return aggLoss.data;
}

function batch<T>(array: T[], batchSize: number = 64) {
    const batches = [];
    for (let i = 0; i < array.length; i += batchSize) {
        batches.push(array.slice(i, i + batchSize));
    }

    return batches;
}

function clipGradient(grad: number, min: number, max: number) {
    if (grad < min) return min;
    if (grad > max) return max;
    return grad;
}


function createEmbeddings(tkn: ITokenizerLite, embSize: number) {
    const embeddings: Value[][] = [];
    for(let i = 0; i < tkn.vocab.length; i++) {
        const emb: Value[] = []
        for(let j = 0; j < embSize; j++) {
            emb.push(new Value(Math.random()))
        }
        embeddings.push(emb)
    }

    return embeddings;
}

function shuffle<T>(array: T[]) {
    let currentIndex = array.length,  randomIndex;
  
    while (currentIndex > 0) {
        randomIndex = Math.floor(Math.random() * currentIndex);
        currentIndex--;
  
      [array[currentIndex], array[randomIndex]] = [
        array[randomIndex], array[currentIndex]];
    }
  
    return array;
}

function split<T>(array: T[], ratio: number = .8) {
    const splitIdx = Math.floor(array.length * ratio);
    const train = array.slice(0, splitIdx);
    const valid = array.slice(splitIdx);
    return [train, valid];
}

export default MLP
