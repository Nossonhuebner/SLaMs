import Hebrew from '../raw_hebrew'
import English from '../raw_english'
import { DumbTokenizer, clean } from './util';
export function SlammyGrammy() {
        const cleaned = clean(Hebrew)

        const cleanE = clean(English);
        window.tkn = new DumbTokenizer(cleaned);
        window.tkn2 = new DumbTokenizer(cleanE);

        console.log(cleanE)
        return (
                <div>
                        <h1>SlammyGrammy</h1>
                </div>
        );
}

declare global {
    interface Window {
        tkn: DumbTokenizer;
        tkn2: DumbTokenizer;

    }
}