// import Hebrew from '../../raw_hebrew'
import English from '../../raw_english'
import { DumbTokenizer, clean } from '../util';
import * as tf from '@tensorflow/tfjs';
import DisplayTable from './DisplayTable';
import { buildGrid, calculateLoss, generateWords } from './util';

export function SlammyGrammy() {
    const cleaned = clean(English);
    const tkn = new DumbTokenizer(cleaned);

    const tokens = tkn.tokenize(cleaned);
    const x = tokens.slice(0, tokens.length - 1);
    const y = tokens.slice(1, tokens.length);

    const grid = buildGrid(x, y, tkn);
    const normalizedGrid = buildGrid(x, y, tkn, true);

    window.tkn = tkn;
    window.txt = cleaned;
    window.tensor = grid;
    window.tf = tf;

    console.log(calculateLoss(normalizedGrid, x, y, tkn))

    return (
        <div>
            <h1>Slam Grams</h1>
            counts: <DisplayTable grid={grid} intToChar={tkn.reverseMap} />
            normalized per row as probabilities: <DisplayTable grid={normalizedGrid} intToChar={tkn.reverseMap} />
            {generateWords(normalizedGrid, tkn, 100).map((word, i) => <div key={i}>{word}</div>)}
        </div>
    );
}



declare global {
    interface Window {
        tkn: DumbTokenizer;
        txt: string;
        tensor: number[][];
        tf: typeof tf;
    }
}