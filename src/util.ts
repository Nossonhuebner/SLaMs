const constants = {
    pos: '{_pos_}',
    prsh_p: '{_prsh_p_}',
    prsh_s: '{_prsh_}',
    sfr: '{_sfr_}',
    mid: '{_mid_}',
    cap: '{_cap_}',
}

export function clean(str: string) {
    if (str.match(/[א-ת]/)) {
        return cleanHebrew(str);
    } else {
        return cleanEnglish(str);
    }
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
        this.tokenize(str);
    }

    decode(tokens: number[]): string {
        return tokens.map(t => this.reverseMap[t]).join('');
    }

    encode(str: string): number[] {
        return this.tokenize(str);
    }

    private tokenize(str: string): number[] {
        const chars = str.split('')
        const result: number[] = [];
        for(let i = 0; i < chars.length; i++) {
            const char = chars[i];
            if (char == '{') {
                const end = str.indexOf('}', i);
                const token = str.substring(i, end + 1);
                this.addToMap(token);
                i += token.length - 1;
            } else {
                this.addToMap(char[i]);
                const charIndex = this.map.get(char);
                if (typeof charIndex !== 'undefined') {
                    result.push(charIndex);
                }
            }
        }
        return result;
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



function cleanHebrew(str: string) {
    const posuk = new RegExp(/[א-ת]{1,2},[א-ת]{1,2}/, 'gi');
    const parsha_p = new RegExp(/\s{2}{פ}/, 'gi');
    const parsha_s = new RegExp(/\s{2}{ס}/, 'gi');
    const sefer = new RegExp(/\s{2}{ש}/, 'gi');
    const mid = ';'
    const hyphen = new RegExp(/-/, 'gi');
    const comma = new RegExp(/,/, 'gi');
    const newLine = new RegExp(/\n/, 'gi');

    let result = str.replaceAll(posuk, constants.pos);
    result = result.replaceAll(parsha_p, constants.prsh_p);
    result = result.replaceAll(parsha_s, constants.prsh_s);
    result = result.replaceAll(sefer, constants.sfr);
    result = result.replaceAll(mid, mid);
    result = result.replaceAll(hyphen, ' ');
    result = result.replaceAll(comma, '');
    result = result.replaceAll(newLine, '');
    result = result.replace(/\s+/g, ' ');

    return result;
}

function cleanEnglish(str: string) {
    const chapter = new RegExp(/Chapter\s\d{1,3}/, 'gi');
    const posuk = new RegExp(/\d{1,3}/, 'gi');

    let result = str.replaceAll(chapter, ' ');
    result = result.replaceAll(posuk, constants.pos);
    result = result.replaceAll('{P}', constants.prsh_p);
    result = result.replaceAll('{S}', constants.prsh_s);
    result = result.replaceAll('{B}', constants.sfr);

    result = result.replace(/\s+/g, ' ');
    result = result.split('').map(c => {
        const lower = c.toLowerCase();
        return c == lower ? c : `${constants.cap}${lower}`
    }).join('')

    return result;
}

