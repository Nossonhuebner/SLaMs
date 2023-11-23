import { useState } from "react";
import { DumbTokenizer, clean } from "../../util";
import English from '../../../raw_english'
import { Value } from "../../util/engine";

function MLP() {
    const cleaned = clean(English);
    const [n, setN] = useState<number>()
    const tkn = new DumbTokenizer(cleaned);

    tkn.map


    return null
}

function createEmbeddings(tkn: DumbTokenizer, size: number) {
    const embeddings = Object.keys(tkn.map).map((token) => {
        let embedding = new Array(size);
        embedding = embeddings.map(e => new Value(Math.random()))
        return embedding;
    })
    return embeddings;
}

export default MLP
