package com.example.tweetsclassifier.deeplearning;

import org.deeplearning4j.iterator.CnnSentenceDataSetIterator;
import org.deeplearning4j.iterator.LabeledSentenceProvider;
import org.deeplearning4j.iterator.provider.CollectionLabeledSentenceProvider;
import org.deeplearning4j.iterator.provider.FileLabeledSentenceProvider;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class Classification {

    private static final long SEED = 239;
    private static final double LEARNING_RATE = 0.01;
    private static final int SENTENCE_LENGTH = 15;
    private static final int BATCH_SIZE = 100;
    private static final int VECTOR_SIZE = 400;
    private static final int DEPTH = 300;
    private static final int EPOCH = 10;
    private static final Logger logger = LoggerFactory.getLogger(Classification.class);

    public static void train(Word2Vec w2v) throws IOException {

        // Configuration of neural network
        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.RELU)
                .activation(Activation.LEAKYRELU)
                .updater(new Adam(LEARNING_RATE))
                .convolutionMode(ConvolutionMode.Same)
                .l2(0.0001)
                .graphBuilder()
                .addInputs("input")
                .addLayer("cnn3", new ConvolutionLayer.Builder()
                        .kernelSize(3, VECTOR_SIZE)
                        .stride(1, VECTOR_SIZE)
                        .nOut(DEPTH)
                        .build(), "input")
                .addLayer("cnn4", new ConvolutionLayer.Builder()
                        .kernelSize(4, VECTOR_SIZE)
                        .stride(1, VECTOR_SIZE)
                        .nOut(DEPTH)
                        .build(), "input")
                .addLayer("cnn5", new ConvolutionLayer.Builder()
                        .kernelSize(5, VECTOR_SIZE)
                        .stride(1, VECTOR_SIZE)
                        .nOut(DEPTH)
                        .build(), "input")
                .addVertex("merge", new MergeVertex(), "cnn3", "cnn4", "cnn5")
                .addLayer("globalPool", new GlobalPoolingLayer.Builder()
                        .poolingType(PoolingType.MAX)
                        .dropOut(0.5)
                        .build(), "merge")
                .addLayer("out", new OutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nOut(2)    //2 classes: positive or negative
                        .build(), "globalPool")
                .setOutputs("out")
                .setInputTypes(InputType.convolutional(SENTENCE_LENGTH, VECTOR_SIZE, 1))
                .build();

        ComputationGraph model = new ComputationGraph(config);
        model.init();

        logger.info("Loading words vectors...");

        DataSetIterator trainIter = getDataSetIterator(w2v);

        logger.info("Training..");
        model.addListeners(new ScoreIterationListener(1));

        model.fit(trainIter, EPOCH);

        Evaluation evaluation = model.evaluate(trainIter);
        System.out.println(evaluation.stats());

        logger.info("Saving model...");

        model.save(new File("saved assets/classification_model"));
    }

    public static double test(Word2Vec w2v, String sentence) throws IOException {

        ComputationGraph model = ComputationGraph.load(new File("saved assets/classification_model"), true);

        DataSetIterator iterator = getDataSetIterator(w2v);
        INDArray features = ((CnnSentenceDataSetIterator) iterator).loadSingleSentence(sentence);

        INDArray result = model.outputSingle(features);
        List<String> labels = iterator.getLabels();

        List<Double> scores = new ArrayList<>();

        for (int i = 0; i < labels.size(); i++) {
            System.out.println(labels.get(i) + ": " + result.getDouble(i));
            scores.add(result.getDouble(i));
        }

        return scores.get(1) - scores.get(0);

    }

    public static DataSetIterator getDataSetIterator(Word2Vec w2v) {
        logger.info("Dataset reading...");

        LabeledSentenceProvider labeledSentenceProvider = readSentencesFromFiles("saved assets/Classification Dataset");

        return new CnnSentenceDataSetIterator.Builder(CnnSentenceDataSetIterator.Format.CNN2D)
                .sentenceProvider(labeledSentenceProvider)
                .wordVectors(w2v)
                .minibatchSize(BATCH_SIZE)
                .maxSentenceLength(SENTENCE_LENGTH)
                .useNormalizedWordVectors(false)
                .build();
    }

    public static CollectionLabeledSentenceProvider readSentencesFromFiles(String directory) {

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
                    sentences.add(currLine);
                    labels.add(Integer.toString(fileIndex));
                }

            } catch (IOException e) {
                logger.error(e.getMessage());
            }

            fileIndex++;

        }

        return new CollectionLabeledSentenceProvider(sentences, labels);

    }

}
