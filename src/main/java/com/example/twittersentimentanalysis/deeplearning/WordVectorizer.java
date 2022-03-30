package com.example.twittersentimentanalysis.deeplearning;

import org.deeplearning4j.bagofwords.vectorizer.TfidfVectorizer;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.evaluation.classification.Evaluation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

public class WordVectorizer {

    private static final Logger logger = LoggerFactory.getLogger(WordVectorizer.class);
    private static final double SAMPLING = 1e-4;
    private static final int WORD_FREQUENCY = 7;
    private static final int LAYER_SIZE = 400;
    private static final int ITERATION = 3;
    private static final long SEED = 185;
    private static final int WINDOW_SIZE = 5;
    private static final double LEARNING_RATE = 0.01;
    private static final int NEGATIVE_SAMPLE = 30;
    private static final int BATCH_SIZE = 1000;

    private static List<String> stopWordsReading() {

        List<String> stopwords = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader("saved assets/Sentences Collection/stopwords"))) {

            String currLine;

            while ((currLine = reader.readLine()) != null) {
                stopwords.add(currLine);
            }

        } catch (IOException e) {
            e.printStackTrace();
        }

        return stopwords;

    }

    public static Word2Vec train() throws IOException {

        logger.info("Dataset reading...");
        SentenceIterator iterator = new LineSentenceIterator(new File("saved assets/Sentences Collection/sentences.txt"));

        iterator.setPreProcessor(new SentencePreProcessor() {
            @Override
            public String preProcess(String s) {
                String text = DataProcessing.stringProcess(s);
                return text;
            }
        });

        TokenizerFactory token = new DefaultTokenizerFactory();
        token.setTokenPreProcessor(new CommonPreprocessor());

        // Stop words
        List<String> stopwords = stopWordsReading();

        logger.info("Word2Vec....");
        Word2Vec w2v = new Word2Vec.Builder()
                .sampling(SAMPLING)
                .minWordFrequency(WORD_FREQUENCY)
                .layerSize(LAYER_SIZE) // size of the vector
                .iterations(ITERATION)
                .seed(SEED)
                .windowSize(WINDOW_SIZE)
                .iterate(iterator)
                .learningRate(LEARNING_RATE)
                .tokenizerFactory(token)
                .batchSize(BATCH_SIZE)
                .stopWords(stopwords)
                .build();

        logger.info("Fitting");
        w2v.setSentenceIterator(iterator);
        w2v.fit();

        logger.info("Saving model...");
        WordVectorSerializer.writeFullModel(w2v, "saved assets/w2v.vec");
//        WordVectorSerializer.writeWord2VecModel(w2v, "saved assets/w2v.dat");

        return w2v;
    }

    public static void test(Word2Vec w2v, String word) {

        System.out.println(w2v.getWordVectorMatrix(word));

        logger.info("Closest Words:");
        Collection<String> list = w2v.wordsNearest(word, 15);

        for (String w : list) {
            System.out.println(w);
            double cosSim = w2v.similarity(word, w);
            System.out.println(cosSim);
        }

    }

}
