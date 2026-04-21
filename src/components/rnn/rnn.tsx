import React, { useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import { Tokenizer, createDataset } from '../../util/tok2';
import { genesisChars } from '../../util/genesis';

const RNN: React.FC = () => {
    const [model, setModel] = useState<tf.LayersModel | null>(null);
    const [tokenizer, setTokenizer] = useState<Tokenizer | null>(null);
    const [trainingStatus, setTrainingStatus] = useState<string>('');
    const [generatedText, setGeneratedText] = useState<string>('');
    const contextLength = 3;
    const SAMPLE_SIZE = 5000;

    useEffect(() => {
        // Set backend to CPU
        tf.setBackend('cpu').then(() => {
            console.log('Using CPU backend:', tf.getBackend());

            // Initialize tokenizer
            const tok = new Tokenizer(genesisChars);
            setTokenizer(tok);

            // Create the model
            const createModel = () => {
                const model = tf.sequential();

                model.add(tf.layers.embedding({
                    inputDim: tok.vocab.length,
                    outputDim: 8,                // small embedding for character-level
                    inputLength: contextLength,
                }));
                model.add(tf.layers.simpleRNN({
                    units: 64,
                    returnSequences: false,
                }));
                model.add(tf.layers.dense({
                    units: tok.vocab.length,
                    activation: 'softmax',
                }));
                model.compile({
                    optimizer: 'adam',
                    loss: 'sparseCategoricalCrossentropy',
                    metrics: ['accuracy'],
                });

                return model;
            };

            setModel(createModel());
        });
    }, []);

    const trainModel = async () => {
        if (!model || !tokenizer) return;

        // Create training data (sampled to avoid freezing the browser)
        const dataset = createDataset(genesisChars.slice(0, SAMPLE_SIZE), tokenizer, contextLength);

        // Convert to tensors (embedding layer expects 2D int token input)
        const inputs = tf.tensor2d(
            dataset.map(d => d.input),
            [dataset.length, contextLength],
            'int32'
        );
        const outputs = tf.tensor1d(
            dataset.map(d => d.output),
            'int32'
        );

        // Train the model
        try {
            await model.fit(inputs, outputs, {
                epochs: 5,
                batchSize: 32,
                callbacks: {
                    onEpochEnd: (epoch, logs) => {
                        setTrainingStatus(
                            `Epoch ${epoch + 1}/5 - loss: ${logs?.loss.toFixed(4)} - accuracy: ${logs?.acc.toFixed(4)}`
                        );
                    }
                }
            });
            setTrainingStatus('Training complete!');
        } catch (error) {
            setTrainingStatus('Error during training: ' + error);
            console.error('Training error:', error);
        }

        // Clean up tensors
        inputs.dispose();
        outputs.dispose();
    };

    const generateText = async (length: number = 100) => {
        if (!model || !tokenizer) return;

        let currentSequence = tokenizer.encode(genesisChars.slice(0, contextLength));
        let generatedString = '';

        for (let i = 0; i < length; i++) {
            // Predict next character
            const input = tf.tensor2d(
                [currentSequence.slice(-contextLength)],
                [1, contextLength],
                'int32'
            );
            const prediction = model.predict(input) as tf.Tensor;
            const probabilities = await prediction.data();
            
            // Sample from the predicted probabilities
            const nextIndex = tf.multinomial(tf.tensor1d(probabilities), 1).dataSync()[0];
            
            // Convert to character and append
            const nextChar = tokenizer.decode([nextIndex]);
            generatedString += nextChar;

            // Update sequence
            currentSequence = [...currentSequence.slice(1), nextIndex];

            // Clean up tensors
            input.dispose();
            prediction.dispose();
        }

        setGeneratedText(generatedString);
    };

    return (
        <div style={{ padding: '20px' }}>
            <h2>Simple RNN Text Generator</h2>
            <div style={{ marginBottom: '20px' }}>
                <button 
                    onClick={() => trainModel()}
                    disabled={!model || !tokenizer}
                    style={{ marginRight: '10px' }}
                >
                    Train Model
                </button>
                <button 
                    onClick={() => generateText()}
                    disabled={!model || !tokenizer}
                >
                    Generate Text
                </button>
            </div>
            
            {trainingStatus && (
                <div style={{ marginBottom: '20px' }}>
                    <h3>Training Status:</h3>
                    <p>{trainingStatus}</p>
                </div>
            )}
            
            {generatedText && (
                <div>
                    <h3>Generated Text:</h3>
                    <p style={{ whiteSpace: 'pre-wrap' }}>{generatedText}</p>
                </div>
            )}
        </div>
    );
};

export default RNN; 