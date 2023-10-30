import Hebrew from '../raw_hebrew'
import English from '../raw_english'
import { DumbTokenizer, clean } from './util';

import { DataGrid } from '@mui/x-data-grid';


export function SlammyGrammy() {
    // const cleaned = clean(Hebrew)
    // window.tkn = new DumbTokenizer(cleaned);

    const cleaned = clean(English);
    const tkn = new DumbTokenizer(cleaned);
    window.tkn = tkn;
    window.txt = cleaned;

    const tokens = tkn.tokenize(cleaned);
    const x = tokens.slice(0, tokens.length - 1);
    const y = tokens.slice(1, tokens.length);
    buildDict(x, y)


    return (
        <div>
            <h1>Slam Grams</h1>
            <DisplayGrid x={x} y={y} tkn={tkn} />
        </div>
    );
}

function DisplayGrid({ x, y, tkn }: { x: string[], y: string[], tkn: DumbTokenizer}) {
    const grid = buildGrid(x, y, tkn)

    const rows = grid.map((row, i) => {
        const rrow: any = { id: i, name: tkn.reverseMap[i] }
        row.forEach((v, j) => {
            rrow[j.toString()] = v.toString();
        })
        return rrow;
    });

    const cols = grid[0].map((_, i) => {
        return { field: i.toString(), headerName: tkn.reverseMap[i] }
    })
    return (
        <DataGrid rows={rows} columns={[{ field: 'name', headerName: 'name' }, ...cols]} style={{ background: 'wheat' }}/>
    );

}

function buildDict(x: string[], y: string[]) {
    const dict = new Map<string, number>();
    for (let i = 0; i < x.length; i++) {
        const key = x[i] + y[i];
        if (dict.has(key)) {
            dict.set(key, dict.get(key)! + 1);
        } else {
            dict.set(key, 1);
        }
    }
    window.dict = dict;
}

function buildGrid(x: string[], y: string[], tkn: DumbTokenizer) {
    const tensor = [];

    for (let i = 0; i < tkn.map.size; i++) {
        const row: number[] = []
        tensor.push(row)
        for (let j = 0; j < tkn.map.size; j++) {
            row.push(0)
        }
    }

    for (let i = 0; i < x.length; i++) {
        const xIdx = tkn.map.get(x[i])!
        const yIdx = tkn.map.get(y[i])!
        tensor[xIdx][yIdx]++
    }
    window.tensor = tensor;
    return tensor;
}

declare global {
    interface Window {
        tkn: DumbTokenizer;
        txt: string;
        dict: Map<string, number>;
        tensor: number[][];
    }
}