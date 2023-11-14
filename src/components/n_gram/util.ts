import * as tf from '@tensorflow/tfjs';
import { DumbTokenizer } from "../../util";

export function generateWords(grid: number[][], tkn: DumbTokenizer, n: number) {
    const words = [];
    for (let i = 0; i < n; i++) {
        words.push(generateWord(grid, tkn));
    }
    return words;
}

function generateWord(grid: number[][], tkn: DumbTokenizer) {
    const word = [];
    let next = sampleRow(grid[0]);
    while (next !== 0) {
        word.push(next);
        next = sampleRow(grid[next]);
    }
    return word.map(i => tkn.reverseMap[i]).join('');
}

function sampleRow(row: number[]) {
    return tf.multinomial(row, 1, undefined, true).arraySync()[0] as number;
}
 
export function buildGrid(x: string[], y: string[], tkn: DumbTokenizer, normalized?: boolean) {
    let grid = [];

    for (let i = 0; i < tkn.map.size; i++) {
        const row: number[] = []
        grid.push(row)
        for (let j = 0; j < tkn.map.size; j++) {
            row.push(0)
        }
    }

    for (let i = 0; i < x.length; i++) {
        const xIdx = tkn.map.get(x[i])!
        const yIdx = tkn.map.get(y[i])!
        grid[xIdx][yIdx]++
    }

    if (normalized) {
        const tensor = tf.tensor(grid);
        grid = tensor.div(tensor.sum(1, true)).arraySync() as number[][];
    }
    return grid;
}

export function calculateLoss(grid: number[][], x: string[], y: string[], tkn: DumbTokenizer) {
    const intX = x.map(char => tkn.map.get(char) as number);
    const intY = y.map(char => tkn.map.get(char) as number);

    let logLikelihood = 0;
    let count = 0
    for (let i = 0; i < intX.length; i++) {
        const row = grid[intX[i]];
        const col = row[intY[i]];
        // console.log(`${tkn.reverseMap[intX[i]]}${tkn.reverseMap[intY[i]]}: ${col}`)
        logLikelihood += Math.log(col);
        count++;
    }
    logLikelihood = -logLikelihood;
    return [logLikelihood, count, logLikelihood / count];

}
