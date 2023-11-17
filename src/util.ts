
// const constants = {
//     pos: '{_pos_}',
//     prsh_p: '{_prsh_p_}',
//     prsh_s: '{_prsh_s_}',
//     sfr: '{_sfr_}',
//     mid: '{_mid_}',
//     cap: '{_cap_}',
//     start: '{_start_}',
//     end: '{_end_}',
// }

export function clean(str: string) {
    if (str.match(/[א-ת]/)) {
        return simpleCleanH(str);
    } else {
        return simpleCleanE(str);
    }
}

export function simpleCleanH(str: string) {
    const posuk = new RegExp(/[א-ת]{1,2},[א-ת]{1,2}/, 'gi');
    let result = str.replaceAll(posuk, '');
    result = result.replaceAll(/[^א-ת]/g, " ");
    result = result.replaceAll('-', ' ');
    const miscRemoval = new RegExp(/[\n,.;]/, 'gi');
    result = result.replaceAll(miscRemoval, '');
    result = result.replaceAll(/\s+/g, ' ');
    result = result.trim();
    result = result.replaceAll(' ', '<S><E>');
    return `<S>${result}<E>`
}

function simpleCleanE(str: string) {
    let result = str.toLowerCase();
    // result = result.split('').map(c => {
    //     const lower = c.toLowerCase();
    //     return c == lower ? c : `${constants.cap}${lower}`
    // }).join('')

    const miscRemoval = new RegExp(/[?,.;!\':\[\]]/, 'gi');
    
    result = result.replaceAll(miscRemoval, '');
    result = result.replaceAll('-', ' ');

    const special = new RegExp(/<[a-zA-Z]{2}>/, 'gi');
    result = result.replaceAll(special, '');
    result = result.replaceAll(/\s+/g, ' ');
    result = result.trim();
    result = result.replaceAll(' ', '*');
    return result
}



interface Tokenizer {
    encode(str: string): (number|undefined)[];
    decode(tokens: number[]): string;
    tokenize(str: string): string[];
    addToMap(str: string): void;
    buildDataSet(str: string): void;
    

}


export class DumbTokenizer {
    map: Map<string, number>;
    reverseMap: string[];
    tokenIndex: number;
    counts: {[key: string]: number} = {};
    constructor(str: string) {
        this.tokenIndex = 0;
        this.map = new Map();
        this.reverseMap = [];
        this.buildDataSet(str);
    }

    buildDataSet(str: string) {
        const tokens = this.tokenize(str);
        tokens.sort();
        tokens.forEach(t => this.addToMap(t));
    }

    decode(tokens: number[]): string {
        const me = this;
        return tokens.map(t => me.reverseMap[t]).join('');
    }

    encode(str: string): (number|undefined)[] {
        const tokens = this.tokenize(str);
        return tokens.map(t => this.map.get(t));
    }


    tokenize(str: string): string[] {
        const chars = str.split('');
        const tokens: string[] = [];
        for(let i = 0; i < chars.length; i++) {
            const char = chars[i];
            if (char == '{') {
                const end = str.indexOf('}', i);
                const token = str.substring(i, end + 1);
                tokens.push(token);
                i += token.length - 1;
            } else if (char == '<') {
                const end = str.indexOf('>', i);
                const token = str.substring(i, end + 1);
                tokens.push(token);
                i += token.length - 1;
            } else {
                tokens.push(char);
            }
        }
        return tokens;
    }

    addToMap(str: string) {
        if (!this.map.has(str)) {
            this.counts[str] = 0;
            this.map.set(str, this.tokenIndex);
            this.reverseMap.push(str);
            this.tokenIndex++;
        }
        this.counts[str]++;
    }

}



// function cleanHebrew(str: string) {
//     const posuk = new RegExp(/\s[א-ת]{1,2},[א-ת]{1,2}/, 'gi');
//     const parsha_p = new RegExp(/\s{2}{פ}/, 'gi');
//     const parsha_s = new RegExp(/\s{2}{ס}/, 'gi');
//     const sefer = new RegExp(/{ספר}/, 'gi');
//     const mid = new RegExp(/[:;]/, 'gi');
//     const hyphen = new RegExp(/-/, 'gi');
//     const miscRemoval = new RegExp(/[\n,.]/, 'gi');

//     let result = str.replaceAll(posuk, constants.pos);
//     result = result.replaceAll(parsha_p, constants.prsh_p);
//     result = result.replaceAll(parsha_s, constants.prsh_s);
//     result = result.replaceAll(sefer, constants.sfr);
//     result = result.replaceAll(mid, constants.mid);
//     result = result.replaceAll(hyphen, ' ');
//     result = result.replaceAll(miscRemoval, '');
//     result = result.replace(/\s+/g, ' ');

//     return result;
// }

// function cleanEnglish(str: string) {
//     const chapter = new RegExp(/Chapter\s\d{1,3}/, 'gi');
//     const posuk = new RegExp(/\d{1,3}/, 'gi');

//     let result = str.replaceAll(chapter, ' ');
//     result = result.replaceAll(posuk, constants.pos);
//     result = result.replaceAll('{P}', constants.prsh_p);
//     result = result.replaceAll('{S}', constants.prsh_s);
//     result = result.replaceAll('{B}', constants.sfr);

//     result = result.replace(/\s+/g, ' ');
//     result = result.split('').map(c => {
//         const lower = c.toLowerCase();
//         return c == lower ? c : `${constants.cap}${lower}`
//     }).join('')

//     return result;
// }

