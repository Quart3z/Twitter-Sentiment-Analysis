package com.example.twittersentimentanalysis.deeplearning;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.deeplearning4j.bagofwords.vectorizer.TfidfVectorizer;

import java.io.File;
import java.io.IOException;

public class WordFrequencyAnalyzer {

    private static final Logger logger = LoggerFactory.getLogger(WordFrequencyAnalyzer.class);

    private static final int WORD_FREQUENCY = 5;

    public static void train() throws IOException {

        logger.info("Reading from file...");
        SentenceIterator iterator = new LineSentenceIterator(new File("saved assets/sentences.txt"));

        TfidfVectorizer vectorizer = new TfidfVectorizer.Builder()
                .setIterator(iterator)
                .setTokenizerFactory(new DefaultTokenizerFactory())
                .setMinWordFrequency(WORD_FREQUENCY )
                .build();

        vectorizer.buildVocab();

        logger.info("Fitting");
        vectorizer.fit();

        WordVectorSerializer.writeVocabCache(vectorizer.getVocabCache(), new File("saved assets/tfidfVectorizer.dat"));

    }

    public static double test(VocabCache vocabCache, String word) {

        logger.info("Initializing tfidfVectorizer");

        org.deeplearning4j.bagofwords.vectorizer.TfidfVectorizer vectorizer = new org.deeplearning4j.bagofwords.vectorizer.TfidfVectorizer.Builder()
                .setVocab(vocabCache)
                .setTokenizerFactory(new DefaultTokenizerFactory())
                .build();

        double result = vectorizer.tfidfWord(word, 1, 1);

        return !Double.isNaN(result) ? result : 0.0;
    }
}
