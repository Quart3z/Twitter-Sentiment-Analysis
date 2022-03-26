package com.example.tweetsclassifier.deeplearning;

import org.deeplearning4j.clustering.cluster.Point;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class SentimentAnalysis {

    private static final Logger logger = LoggerFactory.getLogger(SentimentAnalysis.class);

    public static double sentimentAnalysis(String sentence, Word2Vec w2vModel, List<Point> centroids, VocabCache vocabCache) {

        String[] words = sentence.split(" ");

        double[] sentimentScores = new double[words.length];
        double[][] tfidfScores = new double[1][words.length];

        for (int i = 0; i < words.length; i++) {
            sentimentScores[i] = Clustering.test(w2vModel, centroids, words[i]);
            tfidfScores[0][i] = TfidfVectorizer.tfidfTest(vocabCache, words[i]);
        }

        INDArray m1 = Nd4j.createFromArray(sentimentScores);
        INDArray m2 = Nd4j.createFromArray(tfidfScores);

        logger.info("Comparing overall sentiment score");

        return m2.mmul(m1).getDouble(0);

    }
}
