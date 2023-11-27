import { useMemo, useState } from "react";
import { CharacterTokenizer, ITokenizer, clean } from "../../util";
import English from '../../../raw_english'
import { Value } from "../../util/engine";
import { MLP as Net, softmax } from "../../util/nn";

export function MLP() {
    const cleaned = clean(English);
    const sample = cleaned.slice(0, 2000)
    const tkn = new CharacterTokenizer(sample);
    const embeddingSize = 3;
    const contextLength = 4;

    const embeddings = useMemo(() => createEmbeddings(tkn, embeddingSize), []);
    const net = useMemo(() => new Net(embeddingSize * contextLength, [tkn.vocabulary.length]), []);

    const {training, validation } = createDataset(sample, contextLength, tkn);

    // const [lr, setLr] = useState(0.001);
    // const [loss, setLoss] = useState(0);
    // const [accuracy, setAccuracy] = useState(0);

    // const [epoch, setEpoch] = useState(0);
    for(let i = 0; i < 10; i++) {
        train(net, training, embeddings, 0.01);
        valid(net, validation, embeddings);
    }

    const a = generateWords(10, net, tkn, contextLength, embeddings);
    console.log(a);
}

export function generateWords(n: number, net: Net, tkn: CharacterTokenizer, contextLength: number, embeddings: Value[][]) {
    const words = [];
    for (let i = 0; i < n; i++) {
        words.push(generateWord(net, tkn, contextLength, embeddings));
    }
    return words;
}

function generateWord(net: Net, tkn: CharacterTokenizer, contextLength: number, embeddings: Value[][]) {
    let base = "*".repeat(contextLength);
    const chars = [];
    let next = null;
    while (next != "*") {
        const input = tkn.encode(base) as number[];
        const embs = getEmbeddings(input, embeddings);
        const logits = net.forward(embs.flat());
        const probs = softmax(logits);
        const idx = probs.map(v => v.data).indexOf(Math.max(...probs.map(v => v.data)));
        next = tkn.decode([idx]);
        chars.push(next);
        base = base.slice(1) + next;
    }
    return chars.join('');
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
