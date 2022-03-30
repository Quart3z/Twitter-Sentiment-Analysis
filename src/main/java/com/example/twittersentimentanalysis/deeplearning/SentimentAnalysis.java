package com.example.twittersentimentanalysis.deeplearning;

import org.deeplearning4j.clustering.cluster.Point;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
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
//        WordVectorizer.train();
//        Word2Vec w2v = WordVectorSerializer.readWord2VecModel("saved assets/word2vec.dat");
        Word2Vec w2v = WordVectorSerializer.loadFullModel("saved assets/w2v.vec");
//        WordVectorizer.test(w2v,"sedih");

//        Classification classification = new Classification.Builder()
//                .word2Vec(w2v)
//                .build();
//
//        classification.train();
        ComputationGraph classifier = ComputationGraph.load(new File("saved assets/classification_model"), true);
        List<String> sentences = new ArrayList<>();
        sentences.add("Ya Allah murahkan rezeki ");
        sentences.add("Hilang nyawaku aku tgk");
        sentences.add("Duuhhh,,,,ini apaan sik. Eklusivisme bgt, sama2 pingin shalat aja seolah mereka enggan bercampur dgn yg tdk sepenam https://t.co/aQU2bDOS5e");
//        sentences.add("menyesal nda ambil postpaid maxis dlu");
//        sentences.add("Jae baik banget ya, gitar dia kasihin ke McKay. Bestfriend, best supporter ini mah");
//        sentences.add("GLUCELLA @ Stem Cell terbaik &amp; Jadi RAHASIA Kecantikan kulit anda. MAU CANTIK GAK PERLU MAHAL INFO/ORDER BBM 59EB7F https://t.co/z9MGylJhIe");
//        sentences.add("salam jumaat\n#fridayPrayer (@ Masjid Dato' Hj Kamaruddin in Petaling Jaya, Selangor) https://t.co/5beOWCB4LC");
//        sentences.add("Lelaki kali time tengah gaduh en memang senang tidur ek");
//        sentences.add("@nunaasptn selamat kan jiwa humor");

        for (String s : sentences) {
            double result = sentimentAnalysis(w2v, classifier, s);
            System.out.println(result > 0 ? "Positive" : "Negative" + "\n");
        }


//        double result = Classification.test(w2v, "tiktok diciptakan untuk memutuskan urat malu umat manusia", classifier);
//


//        Clustering.clusterTrain(w2v, 2, 30);
//        List<Point> centroids = SerializationUtils.readObject(new File("saved assets/cluster.dat"));
////        TfidfVectorizer.tfidfTrain();
//        VocabCache vocabCache = WordVectorSerializer.readVocabCache(new File("saved assets/tfidfVectorizer.dat"));
////

////
//        for (String s : sentences) {
//            System.out.println(SentimentAnalysis.sentimentAnalysis(s, w2v, centroids, vocabCache));
//        }

    }

}
