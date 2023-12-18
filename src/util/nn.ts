import { Value } from "./engine";

export class Neuron {
    weights: Value[];
    bias: Value;
    nonLinear: boolean;
    
    constructor(nInputs: number, nonLinear: boolean, scale?: number) {
        this.weights  = []
        for (let i = 0; i < nInputs; i++) {
            const val = scale ? randomNormal(0, scale) : Math.random()
            this.weights.push(new Value(val));
        }
        this.bias = new Value(0);
        this.nonLinear = nonLinear;
    }

    forward(inputs: (Value | number)[]) {
        let sum = new Value(0);
        for (let i = 0; i < inputs.length; i++) {
            sum = sum.plus(this.weights[i].times(inputs[i]));
        }
        sum = sum.plus(this.bias);
        return sum;
    }

    get parameters() {
        return this.weights.concat(this.bias);
    }
}

export interface ILayer {
    forward(inputs: (Value | number)[]): Value[];
    parameters: Value[];
}

export class Layer {
    neurons: Neuron[];
    constructor(nInputs: number, nOutputs: number, nonLinear: boolean) {
        this.neurons = [];

        const variance = 2 / (nInputs + nOutputs);
        const scale = Math.sqrt(variance);

        for (let i = 0; i < nOutputs; i++) {
            this.neurons.push(new Neuron(nInputs, nonLinear, scale));
        }   1
    }

    forward(inputs: (Value | number)[]) {
        return this.neurons.map(neuron => neuron.forward(inputs));
    }

    get parameters() {
        return this.neurons.map(neuron => neuron.parameters).flat();
    }
}
export const LinearLayer = Layer;


export class MLP {
    layers: Layer[];
    constructor(nInputs: number, nOutputs: number[]) {
        this.layers = [];
        const layerConfig = [nInputs, ...nOutputs];
        for (let i = 0; i < layerConfig.length-1; i++) {
            const nIn = layerConfig[i];
            const nOut = layerConfig[i+1];
            const nonLinear = i < layerConfig.length-2;
            this.layers.push(new Layer(nIn, nOut, nonLinear));
        }
    }

    forward(inputs: (Value | number)[]): Value[] {
        let inns = inputs;
        let outs: Value[] = []; // fix this - was done for typing
        for (let i = 0; i < this.layers.length; i++) {
            outs = this.layers[i].forward(inns);
            inns = outs;
        }
        return outs;
    }

    get parameters() {
        return this.layers.map(layer => layer.parameters).flat();
    }
}

// export class EmbeddingLayer implements ILayer {
//     embeddings: Value[];
//     constructor(nInputs: number, nOutputs: number) {
//         this.embeddings = [];
//         for (let i = 0; i < nInputs; i++) {
//             this.embeddings.push(new Value(Math.random()));
//         }
//     }

//     forward(inputs: (Value | number)[]) {
//         return inputs.map(input => this.embeddings[input as number]);
//     }

//     get parameters() {
//         return this.embeddings;
//     }


// }

export class Model {
    layers: ILayer[];
    constructor(layers: ILayer[]) {
        this.layers = layers;
    }

    forward(inputs: (Value | number)[]) {
        let inns = inputs;
        for (let i = 0; i < this.layers.length; i++) {
            inns = this.layers[i].forward(inns);
        }
        return inns;
    }
}

export function softmax(values: Value[]) {
    // subtract maxVal to avoid NaN from exp explosions!
    const maxVal = Math.max(...values.map(v => v.data))

    const exps = values.map(v => v.minus(maxVal).exp())
    const sum = exps.reduce((acc, v) => acc.plus(v), new Value(0))
    return exps.map(v => v.divide(sum))
}

export function negativeLogLikelihood(probs: Value[], target: number) {
    return probs[target].log().neg()
}


function randomNormal(mean: number, stdDev: number) {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return mean + stdDev * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}