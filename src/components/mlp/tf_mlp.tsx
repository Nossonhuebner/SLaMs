import { clean } from "../../util"
import English from '../../../raw_english'
import * as tf from '@tensorflow/tfjs';
import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import { Button, Grid, TextField, Paper, Typography, Box } from "@mui/material";
import './styles.css'
import { createDataset, predictSequences } from "../../util/util";
import { CharTokenizerLite } from "../../util/tokenizers";
import LinearProgress from '@mui/material/LinearProgress';

interface TrainingMetrics {
    epoch: number;
    loss: number;
    accuracy: number;
    valLoss: number;
    valAccuracy: number;
}

interface SerializedWeight {
    shape: number[];
    dtype: 'float32' | 'int32';
    data: number[];
}

const HIDDEN_SIZE = 64;

function TfMlp() {
    // Model parameters
    const [lr, setLr] = useState(0.005);
    const [batchSize, setBatchSize] = useState(640);
    const [contextLength, setContextLength] = useState(4);
    const [embeddingSize, setEmbeddingSize] = useState(30);
    const [sampleSize, setSampleSize] = useState(40000);
    const [epochs, setEpochs] = useState(30);

    // Training state
    const [isTraining, setIsTraining] = useState(false);
    const [trainingMetrics, setTrainingMetrics] = useState<TrainingMetrics[]>([]);
    const [generated, setGenerated] = useState<string[]>([]);

    // Refs
    const modelRef = useRef<tf.LayersModel | null>(null);
    const workerRef = useRef<Worker | null>(null);

    // Data preparation - memoized so identity is stable across renders
    const tokenizer = useMemo(() => new CharTokenizerLite(), []);
    const cleaned = useMemo(() => clean(English), []);

    // Move data processing into an effect to avoid unnecessary recalculations
    const [X, setX] = useState<number[][]>([]);
    const [Y, setY] = useState<number[]>([]);

    useEffect(() => {
        const sample = cleaned.slice(0, sampleSize);
        const tokens = tokenizer.encode(sample);
        const dataset = createDataset(tokens, contextLength);
        setX(dataset.map(item => item.input));
        setY(dataset.map(item => item.output));
    }, [sampleSize, contextLength, cleaned, tokenizer]);

    // Initialize model
    const initModel = useCallback(() => {
        const m = tf.sequential();
        m.add(tf.layers.embedding({
            inputDim: tokenizer.vocab.length,
            outputDim: embeddingSize,
            inputLength: contextLength
        }));
        m.add(tf.layers.flatten());
        m.add(tf.layers.dense({ units: HIDDEN_SIZE, activation: 'relu' }));
        m.add(tf.layers.dense({ units: tokenizer.vocab.length, activation: 'softmax' }));

        // Dispose old model before assigning new one
        if (modelRef.current) {
            modelRef.current.dispose();
        }
        modelRef.current = m;

        // Nominal compile so the local model is usable for inference.
        // The worker uses its own optimizer/lr during training.
        m.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });
    }, [embeddingSize, contextLength, tokenizer]);

    // Initialize model on mount and when architecture parameters change
    useEffect(() => {
        initModel();
    }, [initModel]);

    // Initialize worker
    useEffect(() => {
        workerRef.current = new Worker(
            new URL('../../workers/training.worker.ts', import.meta.url),
            { type: 'module' }
        );

        workerRef.current.onmessage = (e) => {
            if (e.data.type === 'EPOCH_END') {
                setTrainingMetrics(metrics => [...metrics, {
                    epoch: e.data.epoch,
                    loss: e.data.loss,
                    accuracy: e.data.accuracy,
                    valLoss: e.data.valLoss,
                    valAccuracy: e.data.valAccuracy
                }]);
            } else if (e.data.type === 'TRAINING_COMPLETE') {
                if (modelRef.current) {
                    const tensors = (e.data.weights as SerializedWeight[]).map((w) =>
                        tf.tensor(w.data, w.shape, w.dtype)
                    );
                    modelRef.current.setWeights(tensors);
                    tensors.forEach(t => t.dispose());
                }
                setIsTraining(false);
            }
        };

        return () => {
            workerRef.current?.terminate();
            if (modelRef.current) {
                modelRef.current.dispose();
                modelRef.current = null;
            }
        };
    }, []); // Empty dependency array for worker initialization

    const train = () => {
        if (!modelRef.current || !workerRef.current || X.length === 0 || Y.length === 0) return;

        setIsTraining(true);
        setTrainingMetrics([]);

        workerRef.current.postMessage({
            type: 'START_TRAINING',
            X,
            Y,
            vocabSize: tokenizer.vocab.length,
            embeddingSize,
            contextLength,
            hiddenSize: HIDDEN_SIZE,
            epochs,
            batchSize,
            validationSplit: 0.2,
            learningRate: lr,
        });
    };

    const handleGenerate = () => {
        if (!modelRef.current) return;
        predictSequences(tokenizer, modelRef.current, 10, contextLength,
            (str: string) => setGenerated(prev => [...prev, str])
        );
    };

    const dataNotReady = X.length === 0 || Y.length === 0;

    return (
        <Box sx={{ p: 2 }}>
            <Grid container spacing={2}>
                {/* Parameters Section */}
                <Grid item xs={12}>
                    <Paper sx={{ p: 2 }}>
                        <Typography variant="h6" gutterBottom>Model Parameters</Typography>
                        <Grid container spacing={2}>
                            <Grid item xs={6}><TextField label={`Sample Size (max ${cleaned.length.toLocaleString('en-US')})`} value={sampleSize} onChange={e => setSampleSize(parseInt(e.target.value))} /></Grid>
                            <Grid item xs={6}><TextField label="Batch Size" value={batchSize} onChange={e => setBatchSize(parseInt(e.target.value))} /></Grid>


                            <Grid item xs={6}><TextField label="Context Length" value={contextLength} onChange={e => setContextLength(parseInt(e.target.value))} /></Grid>
                            <Grid item xs={6}><TextField label="Embedding Size" value={embeddingSize} onChange={e => setEmbeddingSize(parseInt(e.target.value))} /></Grid>

                            <Grid item xs={6}><TextField label="Learning Rate" value={lr} onChange={e => setLr(parseFloat(e.target.value))} /></Grid>
                            <Grid item xs={6}><TextField label="Epochs" value={epochs} onChange={e => setEpochs(parseInt(e.target.value))} /></Grid>
                        </Grid>
                    </Paper>
                </Grid>

                {/* Training Section */}
                <Grid item xs={12}>
                    <Paper sx={{ p: 2 }}>
                        <Typography variant="h6" gutterBottom>Training</Typography>
                        <Button
                            variant="contained"
                            onClick={train}
                            disabled={isTraining || !modelRef.current || dataNotReady}
                            sx={{ mr: 2 }}
                        >
                            {isTraining ? 'Training...' : 'Train!'}
                        </Button>
                        <Button
                            variant="contained"
                            onClick={handleGenerate}
                            disabled={isTraining || !modelRef.current}
                        >
                            Generate!
                        </Button>

                        {isTraining && <LinearProgress sx={{ mt: 2 }} />}

                        {trainingMetrics.length > 0 && (
                            <Box sx={{ mt: 2 }}>
                                <Typography variant="subtitle1">Latest Metrics (Epoch {trainingMetrics.length}):</Typography>
                                <Typography>Loss: {trainingMetrics[trainingMetrics.length - 1].loss.toFixed(4)}</Typography>
                                <Typography>Accuracy: {(trainingMetrics[trainingMetrics.length - 1].accuracy * 100).toFixed(2)}%</Typography>
                                <Typography>Validation Loss: {trainingMetrics[trainingMetrics.length - 1].valLoss.toFixed(4)}</Typography>
                                <Typography>Validation Accuracy: {(trainingMetrics[trainingMetrics.length - 1].valAccuracy * 100).toFixed(2)}%</Typography>
                            </Box>
                        )}
                    </Paper>
                </Grid>

                {/* Generated Text Section */}
                <Grid item xs={12}>
                    <Paper sx={{ p: 2 }}>
                        <Typography variant="h6" gutterBottom>Generated Text</Typography>
                        {generated.map((g, i) => (
                            <Paper key={i} sx={{ p: 1, mt: 1, bgcolor: 'grey.100' }}>
                                <Typography>{g}</Typography>
                            </Paper>
                        ))}
                    </Paper>
                </Grid>
            </Grid>
        </Box>
    );
}

export default TfMlp;
