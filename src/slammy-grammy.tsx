import Hebrew from '../raw_hebrew'
import { DumbTokenizer, clean, simpleClean } from './util';
export function SlammyGrammy() {
        const cleaned = simpleClean(Hebrew)

        window.tkn = new DumbTokenizer(cleaned);

        return (
                <div>
                        <h1>SlammyGrammy</h1>
                </div>
        );
}

declare global {
    interface Window {
        tkn: DumbTokenizer;
    }
}