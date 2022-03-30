package com.example.twittersentimentanalysis.deeplearning;

import org.deeplearning4j.clustering.cluster.Point;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class SentimentAnalysis {

    private static final Logger logger = LoggerFactory.getLogger(SentimentAnalysis.class);

    // Clustering
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

    public static void main(String[] args) throws IOException {
        WordVectorizer.train();
//        WordVectors w2v = WordVectorSerializer.readWord2VecModel(new File("saved assets/word2vec.dat"));
        Word2Vec w2v = WordVectorSerializer.loadFullModel("saved assets/w2v.vec");
//        WordVectorizer.test(w2v,"sedih");

        Classification classification = new Classification.Builder()
                .word2Vec(w2v)
                .build();

        classification.train();
        ComputationGraph classifier = ComputationGraph.load(new File("saved assets/classification_model"), true);
        List<String> sentences = new ArrayList<>();
        sentences.add("CEWEK CANTIK MASTURBASI SAMPE LEMAS  Gai18 8 min https://t.co/xtPRF8gvQJ");
        sentences.add("@kaesangp APAKAH SAYA FANTASTIS KARENA SERING MINUM F*NTA?");
        sentences.add("Sahabat, yuk isi paket data Indosat Ooredoo-mu melalui #mandirionline atau #mandiriatm. Ada bonus ekstra kuota 5GB https://t.co/CU7x8tGpBs");
        sentences.add("Waktunya Redeem nih  lumayan banget kan cuma modal main hape doang  Jangan lupa download aplikasi Cashpop lalu https://t.co/EERA3Fg5im");
        sentences.add("Memutihkan Ketiak dan Selangkangan Cuma pakai 2 bahan aja Begini Caranya https://t.co/QTbxDuxRUk");
        sentences.add("#JusticeForAudrey keadilan untuk Audrey");

        sentences.add("etikcom Air liur anjing itu najis muhaladoh ( sngt kuat ) dan hrs dihindari. Jk terkena najis air liur anjing, ha https://t.co/ond6heiJmW");
        sentences.add("@izzsani @heunorass Ni dkt wangsa walk ni. Kalau aku dpt ni aku smack down dedua");
        sentences.add("2 hari peti sejuk kasi perengat untong sangat. https://t.co/jsllfCADFk");
        sentences.add("Aku dah la tengah lapar. AARGGHHHH!!");
        sentences.add("Ya allah malas nya aku nak studyyy");

        for (String s : sentences) {
            double result = sentimentAnalysis(w2v, classifier, s);
            System.out.println(result > 0 ? "Positive" : "Negative" + "\n\n");
        }

    }

}
