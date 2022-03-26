package com.example.tweetsclassifier.deeplearning;

import org.deeplearning4j.clustering.cluster.Point;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.clustering.cluster.Cluster;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.common.util.SerializationUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

public class WordVectorizer {

    private static final Logger logger = LoggerFactory.getLogger(WordVectorizer.class);
    private static final double SAMPLING = 1e-4;
    private static final int WORD_FREQUENCY = 5;
    private static final int LAYER_SIZE = 400;
    private static final int ITERATION = 2;
    private static final long SEED = 123;
    private static final int WINDOW_SIZE = 5;
    private static final double LEARNING_RATE = 0.001;
    private static final int NEGATIVE_SAMPLE = 30;
    private static final int BATCH_SIZE = 1000;

    private static Word2Vec w2v;

    public WordVectorizer() {
        w2v = new Word2Vec();
    }

    public static Word2Vec word2VecTraining(boolean isExists) {

        logger.info("Dataset reading...");
        SentenceIterator iterator = new LineSentenceIterator(new File("saved assets/sentences.txt"));

        iterator.setPreProcessor((SentencePreProcessor) s -> {
            DataProcessing dataProcessing = new DataProcessing();

            return dataProcessing.stringProcess(s);
        });

        TokenizerFactory token = new DefaultTokenizerFactory();
        token.setTokenPreProcessor(new CommonPreprocessor());

        if (isExists) {

            logger.info("Reading from file...");

        }
        else {

            logger.info("Model building...");
            w2v = new Word2Vec.Builder()
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
                    .build();

        }

        logger.info("Fitting");
        w2v.setSentenceIterator(iterator);
        w2v.fit();

        logger.info("Saving model...");
        WordVectorSerializer.writeFullModel(w2v, "saved assets/w2v.vec");

        return w2v;
    }

    public static void word2VecTesting(String word1, String word2) {

        System.out.println(w2v.getWordVectorMatrix(word1));

        logger.info("Closest Words:");
        Collection<String> lst = w2v.wordsNearest(word1, 5);
        System.out.println(lst);

        logger.info("Similarities");
        double cosSim = w2v.similarity(word1, word2);
        System.out.println(cosSim);
    }

    public static void main(String[] args) throws IOException {

        w2v = WordVectorSerializer.loadFullModel("saved assets/w2v.vec");
//        Classification.train(w2v);
        double result = Classification.test(w2v, "tiktok diciptakan untuk memutuskan urat malu umat manusia");
//
        System.out.println(result > 0 ? "Positive" : "Negative");


//        word2VecTesting("sayang", "murid");
//        word2VecTraining(false);
//        Clustering.clusterTrain(w2v, 2, 30);
//        List<Point> centroids = SerializationUtils.readObject(new File("saved assets/cluster.dat"));
////        TfidfVectorizer.tfidfTrain();
//        VocabCache vocabCache = WordVectorSerializer.readVocabCache(new File("saved assets/tfidfVectorizer.dat"));
////
//        List<String> sentences = new ArrayList<>();
//        sentences.add("Makian paling pedes yang emak pernah bilang ke gue.\n\nMasih mending piarain anjing, daripada piarain elu.\n\nSeketika https://t.co/pcJI4hmraq");
//        sentences.add("@ustadtengkuzul Siapa juga ya butuh sama elo Zul, kalok gak mau patuhi pemerintah minggat dah dari Indonesia nyusul yg di arab sono");
//        sentences.add("yakin nga nih @fayrouz_id sari buah kurmanya asli? nanti malah kayak keyakinan dia, palsu huwaaa\ncoba deh nanti gue https://t.co/Ds2Dy94w7i");
//        sentences.add("menyesal nda ambil postpaid maxis dlu");
//        sentences.add("Jae baik banget ya, gitar dia kasihin ke McKay. Bestfriend, best supporter ini mah");
//        sentences.add("GLUCELLA @ Stem Cell terbaik &amp; Jadi RAHASIA Kecantikan kulit anda. MAU CANTIK GAK PERLU MAHAL INFO/ORDER BBM 59EB7F https://t.co/z9MGylJhIe");
//        sentences.add("salam jumaat\n#fridayPrayer (@ Masjid Dato' Hj Kamaruddin in Petaling Jaya, Selangor) https://t.co/5beOWCB4LC");
//        sentences.add("Lelaki kali time tengah gaduh en memang senang tidur ek");
//        sentences.add("@nunaasptn selamat kan jiwa humor");
////
//        for (String s : sentences) {
//            System.out.println(SentimentAnalysis.sentimentAnalysis(s, w2v, centroids, vocabCache));
//        }

//
    }


}
