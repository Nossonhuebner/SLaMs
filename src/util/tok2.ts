export type DataItem = {
    input: number[];
    output: number;
}

export function createDataset(sourceStr: string, tokenizer: Tokenizer, contextLength: number) {
    const r = new Array<DataItem>();
    // Add padding at the start
    sourceStr = " ".repeat(contextLength) + sourceStr
    const tokens = tokenizer.encode(sourceStr)
    console.log("Encoded tokens:", tokens.slice(0, 10), "...")

    for (let i = 0; i < tokens.length - contextLength; i++) {
        const input = tokens.slice(i, i + contextLength) as number[];
        const output = tokens[i + contextLength] as number;
        
        // Validate the data
        if (!input.some(x => x === undefined) && output !== undefined) {
            const item = { input, output };
            r.push(item);
        }
    }

    // Log some sample data for debugging
    console.log("Sample training pairs:", r.slice(0, 3))
    
    return r;
}

export class Tokenizer {
    stringToInt: Map<string, number> // maps string to index
    intToStr: string[] // uses indices to map to string
    _specialChar: string
    _endChar: string
    constructor(text: string, endChar: string = ' ', delimiter: string = '') {
        this._specialChar = '*'
        text = text.replaceAll(endChar, this._specialChar )

        const chars = text.split(delimiter)
        const unique = Array.from(new Set(chars)).sort()
        console.log(unique)
        this.stringToInt = new Map<string, number>()
        this._endChar = endChar
        this.intToStr = unique
        for(let i = 0; i < unique.length; i++) {
            this.stringToInt.set(unique[i],  i)
        }
        console.log(this.intToStr)
        console.log(this.stringToInt)
    }

    encode(str: string): number[] {
        str = str.replaceAll(this._endChar, this._specialChar)
        return str.split('').map(char => this.stringToInt.get(char) as number)
    }

    decode(vals: number[]): string {
        return vals.map(v => this.intToStr[v]).join('').replaceAll(this._specialChar, this._endChar)
    }

    get vocab() {
        return this.intToStr;
    }

    get specialChar() {
        return this._specialChar
    }

    get endChar() {
        return this._endChar
    }

    get specialToken() {  
        return this.stringToInt.get(this._specialChar) as number  
    }
}
