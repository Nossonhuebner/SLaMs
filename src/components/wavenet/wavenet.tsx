import { clean } from "../../util"
import English from '../../../raw_english'
import { CharTokenizerLite } from "../../util/tokenizers";
import { createDataset } from "../../util/util";

function WaveNet() {
    const cleaned = clean(English);
    const tokenizer = new CharTokenizerLite();

    const sample = cleaned.slice(0, 100);
    const tokens = tokenizer.encode(sample);
    const dataset = createDataset(tokens, 8);
    console.log(dataset);
    return (
        <div>
        <h1>WaveNet (under construction)</h1>
        </div>
    );
}

export default WaveNet;