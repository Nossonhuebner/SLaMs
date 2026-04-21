/* eslint-disable no-restricted-globals */
import * as tf from '@tensorflow/tfjs';

// Define the message types
interface TrainingMessage {
  type: 'START_TRAINING';
  X: number[][];
  Y: number[];
  vocabSize: number;
  embeddingSize: number;
  contextLength: number;
  hiddenSize: number;
  epochs: number;
  batchSize: number;
  validationSplit: number;
  learningRate: number;
}

interface TrainingProgressMessage {
  type: 'EPOCH_END';
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

interface TrainingCompleteMessage {
  type: 'TRAINING_COMPLETE';
  weights: SerializedWeight[];
}

// Add type declaration for self
declare const self: Worker & {
  onmessage: (e: MessageEvent<TrainingMessage>) => void;
};

self.onmessage = async (e: MessageEvent<TrainingMessage>) => {
  if (e.data.type !== 'START_TRAINING') return;

  const {
    X,
    Y,
    vocabSize,
    embeddingSize,
    contextLength,
    hiddenSize,
    epochs,
    batchSize,
    validationSplit,
    learningRate,
  } = e.data;

  const model = tf.sequential();
  model.add(tf.layers.embedding({
    inputDim: vocabSize,
    outputDim: embeddingSize,
    inputLength: contextLength,
  }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: hiddenSize, activation: 'relu' }));
  model.add(tf.layers.dense({ units: vocabSize, activation: 'softmax' }));
  model.compile({
    optimizer: tf.train.adam(learningRate),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  const xs = tf.tensor2d(X, [X.length, contextLength], 'int32');
  const ys = tf.oneHot(Y, vocabSize);

  try {
    await model.fit(xs, ys, {
      epochs,
      batchSize,
      validationSplit,
      shuffle: true,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          const progress: TrainingProgressMessage = {
            type: 'EPOCH_END',
            epoch,
            loss: logs?.loss ?? 0,
            accuracy: logs?.acc ?? 0,
            valLoss: logs?.val_loss ?? 0,
            valAccuracy: logs?.val_acc ?? 0,
          };
          self.postMessage(progress);
        },
      },
    });

    const weightTensors = model.getWeights();
    const weights: SerializedWeight[] = await Promise.all(
      weightTensors.map(async (w) => ({
        shape: w.shape,
        dtype: w.dtype as 'float32' | 'int32',
        data: Array.from(await w.data()),
      }))
    );
    const complete: TrainingCompleteMessage = { type: 'TRAINING_COMPLETE', weights };
    self.postMessage(complete);
  } finally {
    xs.dispose();
    ys.dispose();
    model.dispose();
  }
};

export {}; // This makes the file a module
