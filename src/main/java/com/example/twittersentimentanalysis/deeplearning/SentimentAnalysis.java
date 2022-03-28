package com.example.twittersentimentanalysis.deeplearning;

import org.deeplearning4j.clustering.cluster.Point;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.List;

public class SentimentAnalysis {

    private static final Logger logger = LoggerFactory.getLogger(SentimentAnalysis.class);

//    public static double sentimentAnalysis_1(String sentence, Word2Vec w2vModel, List<Point> centroids, VocabCache vocabCache) {
//
//        String[] words = sentence.split(" ");
//
//        double[] sentimentScores = new double[words.length];
//        double[][] tfidfScores = new double[1][words.length];
//
//        for (int i = 0; i < words.length; i++) {
//            sentimentScores[i] = Clustering.test(w2vModel, centroids, words[i]);
//            tfidfScores[0][i] = WordFrequencyAnalyzer.test(vocabCache, words[i]);
//        }
//
//        INDArray m1 = Nd4j.createFromArray(sentimentScores);
//        INDArray m2 = Nd4j.createFromArray(tfidfScores);
//
//        logger.info("Comparing overall sentiment score");
//
//        return m2.mmul(m1).getDouble(0);
//
//    }

    public static double sentimentAnalysis(Word2Vec w2v, ComputationGraph classifier, String text) throws IOException {
        Classification classification = new Classification.Builder()
                .word2Vec(w2v)
                .classifier(classifier)
                .text(text)
                .build();

        return classification.test();
    }

}
