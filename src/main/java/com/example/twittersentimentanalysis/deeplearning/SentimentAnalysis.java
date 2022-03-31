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
//        WordVectorizer.train();
//        WordVectors w2v = WordVectorSerializer.readWord2VecModel(new File("saved assets/word2vec.dat"));
//        Word2Vec w2v = WordVectorSerializer.loadFullModel("saved assets/w2v.vec");
//        WordVectorizer.test(w2v,"sedih");

//        Classification classification = new Classification.Builder()
//                .word2Vec(w2v)
//                .build();
//
//        classification.train();
//        ComputationGraph classifier = ComputationGraph.load(new File("saved assets/classification_model"), true);
//        List<String> sentences = new ArrayList<>();
//        sentences.add("Kejadian kira-kira pukul 11.40 pagi itu berlaku apabila kereta jenis Perodua Myvi dikatakan hilang kawalan lalu mas https://t.co/18Ys0kkYwz");
//        sentences.add("Sumpah penat kalau mcm ni je dri dulu !");
//        sentences.add("Ni kalau pergi klinik pastu kena marah lagi ngan dr sebab x gheti nak tido.");
//        sentences.add("@rmolco Candi borobudur , gak di klaim pak? Museum tsunami pak?");
//        sentences.add("@MiharuKenshin Sama level bodoh dgn you know who. Level taksub");
//        sentences.add("enaknya jadi jomblo itu ya bayar makanan atau minuman 2, dapetnya ya 2, paham ?");
//        sentences.add("Kyk gini kok jd Menteri.");
//        sentences.add(" Aduh rindu bae lah. Tapi semalam baru habiskan masa berlari dkt aeon big sbb takut mr diy dah tutup.");
//        sentences.add("Kepala aku nak pecah je ni ya Allah");
//        sentences.add("W pengen mekap tapi gatau caranya, gatau bahannya, mereknya, bagusan yang mana, manfaatnya &amp; blablabla\nKuota habis https://t.co/XsUTcAClPV");
//        sentences.add("Nak tapi tak mampu ");
//
//        sentences.add("Congrats to Governor Nasir Ahmed El-Rufai @elrufai and Deputy Governor Dr. Hadiza Sabuwa Balarabe @DrHadiza. Allah ya taya ku riko. https://t.co/FqzHWSighK");
//        sentences.add("sesiapa mencari tempat tshirt printing sekitar Shah Alam / Klang ? dengan kuantiti yang banyak dan murah ? boleh PM https://t.co/rT0iPJ6Pc");
//        sentences.add("selama ini aku naik bis, baru td dapet pengamen yg genjrengan gitar dan suaranya enak:') jujur terharu");
//        sentences.add("Kalau hadiahkan kat adik mesti dia");
//
//        for (String s : sentences) {
//            double result = sentimentAnalysis(w2v, classifier, s);
//            System.out.println(result > 0 ? "Positive" : "Negative");
//        }

    }

}
