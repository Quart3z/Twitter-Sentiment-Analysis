package com.example.twittersentimentanalysis.deeplearning;

import org.deeplearning4j.iterator.CnnSentenceDataSetIterator;
import org.deeplearning4j.iterator.LabeledSentenceProvider;
import org.deeplearning4j.iterator.provider.CollectionLabeledSentenceProvider;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.KFoldIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.regularization.L2Regularization;
import org.nd4j.linalg.learning.regularization.Regularization;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class Classification {

    private static final Logger logger = LoggerFactory.getLogger(Classification.class);

    // Hyper parameters
    private static final long SEED = 239;
    private static final double LEARNING_RATE = 0.001;
    private static final int SENTENCE_LENGTH = 100;
    private static final int BATCH_SIZE = 300;
    private static final int VECTOR_SIZE = 400;
    private static final int FEATURES_MAPS = 100;
    private static final int EPOCH = 3;
    private static final int OUTPUT = 2;

    private final WordVectors w2v;
    private final ComputationGraph classifier;
    private final String text;

    private Classification(Builder builder) {
        this.w2v = builder.w2v;
        this.classifier = builder.classifier;
        this.text = builder.text;
    }

    public void train() throws IOException {

        // Configuration of neural network
        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(SEED)
                .weightInit(WeightInit.RELU)
                .activation(Activation.LEAKYRELU)
                .updater(new Adam(LEARNING_RATE))
                .convolutionMode(ConvolutionMode.Same)
                .l2(0.00001)
                .graphBuilder()
                .addInputs("input")
                .addLayer("cnn3", new ConvolutionLayer.Builder()
                        .kernelSize(3, VECTOR_SIZE)
                        .stride(1, VECTOR_SIZE)
                        .nIn(1)
                        .nOut(FEATURES_MAPS)
                        .build(), "input")
                .addLayer("cnn4", new ConvolutionLayer.Builder()
                        .kernelSize(4, VECTOR_SIZE)
                        .stride(1, VECTOR_SIZE)
                        .nIn(1)
                        .nOut(FEATURES_MAPS)
                        .build(), "input")
                .addLayer("cnn5", new ConvolutionLayer.Builder()
                        .kernelSize(5, VECTOR_SIZE)
                        .stride(1, VECTOR_SIZE)
                        .nIn(1)
                        .nOut(FEATURES_MAPS)
                        .build(), "input")
                .addVertex("merge", new MergeVertex(), "cnn3", "cnn4", "cnn5")
                .addLayer("globalPool", new GlobalPoolingLayer.Builder()
                        .poolingType(PoolingType.MAX)
                        .dropOut(0.5)
                        .build(), "merge")
                .addLayer("out", new OutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(3 * FEATURES_MAPS)
                        .nOut(OUTPUT)    // 2 classes: positive or negative
                        .build(), "globalPool")
                .setOutputs("out")
                .setInputTypes(InputType.convolutional(SENTENCE_LENGTH, VECTOR_SIZE, 1))
                .build();

        ComputationGraph model = new ComputationGraph(config);
        model.init();

        logger.info("Loading dataset...");

        DataSetIterator trainIter = getDataSetIterator(true);
        DataSetIterator testIter = getDataSetIterator(false);

        logger.info("Training..");
        model.addListeners(new ScoreIterationListener(1));

        model.fit(trainIter, EPOCH);

        logger.info("Evaluation");
        Evaluation evalTrain = model.evaluate(trainIter);
        Evaluation evalTest = model.evaluate(testIter);

        System.out.println(evalTrain.stats());
        System.out.println(evalTest.stats());

        logger.info("Saving model...");
        model.save(new File("saved assets/classification_model"));
    }

    public double test() throws IOException {

        // Features extraction for input sentence
        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        List<String> tokens = tokenizerFactory.create(DataProcessing.stringProcess(text)).getTokens();
        List<String> filteredTokens = new ArrayList<>();

        for (String token : tokens) {
            if (w2v.hasWord(token)) {
                filteredTokens.add(token);
            }
        }

        int outputLength = Math.min(filteredTokens.size(), SENTENCE_LENGTH);
        INDArray features = Nd4j.create(1, 1, outputLength, VECTOR_SIZE);

        for (int b = 0; b < filteredTokens.size() && b < SENTENCE_LENGTH; b++) {
            INDArray vector = w2v.getWordVectorMatrix(filteredTokens.get(b));
            features.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.point(b), NDArrayIndex.all()}, vector);
        }

        // Prediction
        INDArray result = classifier.outputSingle(features);

        String[] labels = new String[]{"Negative", "Positive"};

        List<Double> scores = new ArrayList<>();

        for (int i = 0; i < labels.length; i++) {
            System.out.println(labels[i] + ": " + result.getDouble(i));
            scores.add(result.getDouble(i));
        }

        return scores.get(1) - scores.get(0);

    }

    private DataSetIterator getDataSetIterator(boolean isTrain) {

        LabeledSentenceProvider labeledSentenceProvider = readSentencesFromFiles(isTrain);

        return new CnnSentenceDataSetIterator.Builder(CnnSentenceDataSetIterator.Format.CNN2D)
                .sentenceProvider(labeledSentenceProvider)
                .wordVectors(w2v)
                .minibatchSize(BATCH_SIZE)
                .maxSentenceLength(SENTENCE_LENGTH)
                .useNormalizedWordVectors(false)
                .build();

    }

    private CollectionLabeledSentenceProvider readSentencesFromFiles(boolean isTrain) {

        String directory;
        if (isTrain) {
            logger.info("Training set reading...");
            directory = "saved assets/Classification Dataset/Train";
        } else {
            logger.info("Testing set reading...");
            directory = "saved assets/Classification Dataset/Test";
        }

        File datasetSource = new File(directory);

        String[] files = datasetSource.list();

        List<String> sentences = new ArrayList<>();
        List<String> labels = new ArrayList<>();

        int fileIndex = 0;

        for (String file : files) {

            logger.info("Reading file - " + (fileIndex + 1));

            try (BufferedReader reader = new BufferedReader(new FileReader(directory + "/" + file))) {

                String currLine;

                while ((currLine = reader.readLine()) != null) {
                    sentences.add(DataProcessing.stringProcess(currLine));
                    labels.add(Integer.toString(fileIndex));
                }

            } catch (IOException e) {
                logger.error(e.getMessage());
            }

            fileIndex++;

        }

        return new CollectionLabeledSentenceProvider(sentences, labels);

    }

    public static class Builder {
        private WordVectors w2v;
        private ComputationGraph classifier;
        private String text;

        public Builder word2Vec(WordVectors w2v) {
            this.w2v = w2v;
            return this;
        }

        public Builder classifier(ComputationGraph classifier) {
            this.classifier = classifier;
            return this;
        }

        public Builder text(String sentence) {
            this.text = sentence;
            return this;
        }

        public Classification build() {
            Classification classification = new Classification(this);
            return classification;
        }
    }

}
