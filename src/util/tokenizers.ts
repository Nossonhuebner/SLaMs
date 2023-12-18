export interface ITokenizerLite {
    encode(str: string|string[]): number[];
    decode(vals: number[]): string;
    vocab: string[];
    specialToken: number;
}

export class CharTokenizerLite implements ITokenizerLite {
    map: Map<string, number>
    iToS: string[]
    _specialToken: number
    constructor() {
        const chars = "*abcdefghijklmnopqrstuvwxyz.".split('')
        this.map = new Map<string, number>()
        this.iToS = chars
        for(let i = 0; i < chars.length; i++) {
            this.map.set(chars[i],  i)
        }
        this._specialToken = this.map.get('*') as number
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

    get specialToken() {    
        return this._specialToken
    }
}


export class WordTokenizerLite implements ITokenizerLite {
    map: Map<string, number>
    iToS: string[]
    _specialToken: number
    constructor(str: string) {
        // expects a string with * as spaces
        const words = str.replaceAll('.', '*.').split('*')
        const unique =  Array.from(new Set(words))
        this.map = new Map<string, number>()
        this.iToS = unique
        for(let i = 0; i < unique.length; i++) {
            this.map.set(unique[i],  i)
        }
        this._specialToken = this.map.get('.') as number
    }

    format(str: string): string[] {
       return str.replaceAll('.', '*.').split('*')
    }

    encode(str: string|string[]): number[] {
        // assumes string[] is already split...
        const val = typeof(str) === 'string' ? this.format(str) : str
        return val.map(word => this.map.get(word) as number)
    }

    decode(vals: number[]): string {
        return vals.map(v => this.iToS[v]).join(' ')
    }

    get vocab() {
        return this.iToS;
    }

    get specialToken() {    
        return this._specialToken
    }
}
