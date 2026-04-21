import { useEffect, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Tokenizer } from '../../util/tok2'
import { createDataset } from '../../util/tok2'
import { createModel, predictSequences } from '../../util/util'
import { genesisClean } from '../../util/genesis'

// Adjust parameters for better learning
const CONTEXT_LENGTH = 4     // Increased context
const HIDDEN_SIZE = 128
const EPOCHS = 50
const BATCH_SIZE = 32      // Reduced batch size for better learning
const VALIDATION_SPLIT = 0.1
const MAX_SAMPLES = 1000   // Increased training data
const LEARNING_RATE = 0.002 // Slightly reduced learning rate

export default function TF2() {
    const [tokenizer, setTokenizer] = useState<Tokenizer>()
    const [model, setModel] = useState<tf.Sequential>()
    const [generated, setGenerated] = useState<string[]>([])
    const [isTraining, setIsTraining] = useState(false)
    const [error, setError] = useState<string>("")

    useEffect(() => {
        (async () => {
            try {
                await tf.setBackend('cpu')
                console.log('Backend:', tf.getBackend())

                const tok = new Tokenizer(genesisClean.slice(0, MAX_SAMPLES))
                console.log('Vocab size:', tok.vocab.length)
                setTokenizer(tok)

                // Create and compile model
                const vocabSize = tok.vocab.length
                const mlp = createModel(CONTEXT_LENGTH, vocabSize, HIDDEN_SIZE)

                mlp.compile({
                    optimizer: tf.train.adam(LEARNING_RATE),
                    loss: 'categoricalCrossentropy',
                    metrics: ['accuracy']
                })

                setModel(mlp)
            } catch (err) {
                console.error("Error in initialization:", err)
                setError(`Init error: ${err}`)
            }
        })()
    }, [])

    const train = async () => {
        try {
            if (!tokenizer || !model) {
                console.error("Missing tokenizer or model")
                return
            }
            setIsTraining(true)
            setError("")

            // Create training data
            const dataset = createDataset(genesisClean.slice(0, MAX_SAMPLES), tokenizer, CONTEXT_LENGTH)

            const inputData = dataset.map(d => d.input)
            const outputData = dataset.map(d => d.output)

            // For embedding layer, inputs should be int32
            const inputs = tf.tensor2d(inputData, [dataset.length, CONTEXT_LENGTH], 'int32')
            const outputs = tf.oneHot(
                tf.tensor1d(outputData, 'int32'),
                tokenizer.vocab.length
            ).asType('float32')

            let startTime = Date.now()

            await model.fit(inputs, outputs, {
                epochs: EPOCHS,
                batchSize: BATCH_SIZE,
                validationSplit: VALIDATION_SPLIT,
                shuffle: true,
                callbacks: {
                    onEpochBegin: (epoch) => {
                        console.log(`Starting epoch ${epoch + 1}/${EPOCHS}`)
                        startTime = Date.now()
                    },
                    onEpochEnd: (epoch, logs) => {
                        const timeElapsed = ((Date.now() - startTime) / 1000).toFixed(1)
                        console.log(
                            `Epoch ${epoch + 1}/${EPOCHS} (${timeElapsed}s): ` +
                            `loss = ${logs?.loss.toFixed(4)}, ` +
                            `val_loss = ${logs?.val_loss?.toFixed(4)}, ` +
                            `acc = ${logs?.acc?.toFixed(4)}, ` +
                            `val_acc = ${logs?.val_acc?.toFixed(4)}`
                        )
                    }
                }
            })

            // Clean up tensors
            inputs.dispose()
            outputs.dispose()

            console.log("Training complete")
            setIsTraining(false)
        } catch (err) {
            console.error("Error in training:", err)
            setError(`Training error: ${err}`)
            setIsTraining(false)
        }
    }

    const generate = () => {
        try {
            if (!tokenizer || !model) {
                console.error("Missing tokenizer or model")
                return
            }
            setError("")
            setGenerated([])

            predictSequences(tokenizer, model, 5, CONTEXT_LENGTH,
                (str: string) => {
                    setGenerated(prev => [...prev, str])
                })
        } catch (err) {
            console.error("Error in generation:", err)
            setError(`Generation error: ${err}`)
        }
    }

    return (
        <div>
            <h2>TensorFlow.js MLP Text Generator</h2>
            <button onClick={train} disabled={isTraining}>
                {isTraining ? 'Training...' : 'Train Model'}
            </button>
            <button onClick={generate} disabled={isTraining || !model}>
                Generate Text
            </button>

            {error && (
                <div style={{color: 'red', marginTop: '10px'}}>
                    Error: {error}
                </div>
            )}

            <div style={{marginTop: '20px'}}>
                <h3>Generated Text:</h3>
                {generated.map((text, i) => (
                    <div key={i}>{text}</div>
                ))}
            </div>
        </div>
    )
}
