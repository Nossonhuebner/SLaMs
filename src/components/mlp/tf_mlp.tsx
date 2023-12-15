import { clean } from "../../util"
import English from '../../../raw_english'
import * as tf from '@tensorflow/tfjs';
import { useMemo, useState } from "react";
import { Button, Grid, TextField,Tooltip, IconButton} from "@mui/material";
import InfoIcon from '@mui/icons-material/Info';              
import './styles.css'
import { createDataset, predictSequences } from "../../util/util";
import { CharTokenizerLite } from "../../util/tokenizers";

function TfMlp() {
    const [lr, setLr] = useState(0.001);
    const [batchSize, setBatchSize] = useState(640);
    const [contextLength, setContextLength] = useState(4);  // Length of each input sequence
    const [embeddingSize, setEmbeddingSize] = useState(30);
    const [sampleSize, setSampleSize] = useState(40000);
    const [epochs, setEpochs] = useState(30);

    const tokenizer = new CharTokenizerLite();
    const cleaned = clean(English);
    const sample = cleaned.slice(0, sampleSize);
    const tokens = tokenizer.encode(sample);
    const [generated, setGenerated] = useState<string[]>([]);

    const inputDim = tokenizer.vocab.length
    const dataset = createDataset(tokens, contextLength)
    const X = dataset.map(item => item.input);
    const Y = dataset.map(item => item.output);

    const model = useMemo(() => {
        const m = tf.sequential();

        m.add(tf.layers.embedding({
            inputDim: inputDim, // vocabulary size
            outputDim: embeddingSize, // embedding size
            inputLength: contextLength // context size
        })); // [(batchSize), contextSize] => [(batchSize), contextSize, embeddingSize]

        m.add(tf.layers.flatten()); // [(batchSize), contextSize, embeddingSize] => [(batchSize), contextSize * embeddingSize]

        m.add(tf.layers.dense({ units: 64, activation: 'relu' })); // [(batchSize), contextSize * embeddingSize] => [(batchSize), 64]

        m.add(tf.layers.dense({ units: tokenizer.vocab.length, activation: 'softmax' })); // [(batchSize), 64] => [(batchSize), vocabularySize]

        m.compile({
            optimizer:  tf.train.adam(lr) ,
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });
        m.summary()
        return m;
    }, [inputDim, embeddingSize, contextLength]);

    const oneHotLabels = tf.oneHot(Y, tokenizer.vocab.length);
    const train = () => {
        model.fit(tf.tensor2d(X, [X.length, contextLength]), oneHotLabels, {
            epochs: epochs,
            validationSplit: 0.2,
            batchSize: batchSize,
            shuffle: true,
            callbacks: {
              onEpochEnd: (epoch, logs) => {
                console.log(logs)
                console.log(`Epoch ${epoch + 1}: Loss: ${logs?.loss}, Val_Accuracy: ${logs?.val_acc}`);
              }
            }
        });
    
    }

    const addGenerated = (str: string) => {
        setGenerated(s => [...s, str]);
    }

    return (
        <>
            <Grid container className="inputContainer">
                <Grid item xs={6}><TextField label={`Sample Size (max ${cleaned.length.toLocaleString('en-US')})`} value={sampleSize} onChange={e => setSampleSize(parseInt(e.target.value))} /></Grid>
                <Grid item xs={6}><TextField label="Batch Size" value={batchSize} onChange={e => setBatchSize(parseInt(e.target.value))} /></Grid>
                

                <Grid item xs={6}><TextField label="Context Length" value={contextLength} onChange={e => setContextLength(parseInt(e.target.value))} /></Grid>
                <Grid item xs={6}><TextField label="Embedding Size" value={embeddingSize} onChange={e => setEmbeddingSize(parseInt(e.target.value))} /></Grid>
                
                <Grid item xs={6}><TextField label="Learning Rate" value={lr} onChange={e => setLr(parseFloat(e.target.value))} /></Grid>
                <Grid item xs={6}><TextField label="Epochs" value={epochs} onChange={e => setEpochs(parseInt(e.target.value))} /></Grid>
            </Grid>
            <Button onClick={train}>
                <Tooltip title="Training deets in console">
                    <IconButton>
                    <InfoIcon />
                    </IconButton>
                </Tooltip>
                Train!
            </Button>
            <Button onClick={() => predictSequences(tokenizer ,model, 10, contextLength, addGenerated)}>Generate!</Button>
            {generated.map((g, i) => <div key={i}>{g}</div>)}
        </>
    )
}




export default TfMlp;