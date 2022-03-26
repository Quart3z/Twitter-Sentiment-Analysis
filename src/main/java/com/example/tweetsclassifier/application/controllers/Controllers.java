package com.example.tweetsclassifier.application.controllers;

import com.example.tweetsclassifier.application.Tweet;
import com.example.tweetsclassifier.deeplearning.Clustering;
import com.example.tweetsclassifier.deeplearning.DataProcessing;
import com.example.tweetsclassifier.deeplearning.SentimentAnalysis;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectWriter;
import lombok.Data;
import lombok.SneakyThrows;
import org.apache.tomcat.util.json.JSONParser;
import org.deeplearning4j.clustering.cluster.Cluster;
import org.deeplearning4j.clustering.cluster.Point;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;

import org.nd4j.common.util.SerializationUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Flux;

import javax.naming.ldap.Control;
import java.io.*;
import java.util.ArrayList;
import java.util.List;

@RestController
public class Controllers {

    // Training Materials
    private static List<Point> centroids = new ArrayList<>();
    private final Logger logger = LoggerFactory.getLogger(Control.class);
    private final DataProcessing dataProcessing = new DataProcessing();
    private Word2Vec w2vModel = new Word2Vec();
    private VocabCache vocabCache = new AbstractCache();
    @Autowired
    private WebClient webClient;

    @SneakyThrows
    public Controllers() {
        // files reading
        logger.info("Reading files ...");

        centroids = SerializationUtils.readObject(new File("saved assets/cluster.dat"));
        w2vModel = WordVectorSerializer.loadFullModel("saved assets/w2v.vec");
        vocabCache = WordVectorSerializer.readVocabCache(new File("saved assets/tfidfVectorizer.dat"));

        logger.info("End of files reading");

    }

    @GetMapping(value = "/tweet", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<String> index() {

        logger.info("Started...");

        // Pass to model
        ObjectMapper mapper = new ObjectMapper();
        return this.webClient
                .get()
                .retrieve()
                .bodyToFlux(String.class)
                .map(message -> {

                    System.out.println(message);
                    // Json deserialization
                    try {
                        Tweet tweet = mapper.readValue(message, Tweet.class);
                        String text = dataProcessing.stringProcess(tweet.getText());

                        double score = SentimentAnalysis.sentimentAnalysis(text, w2vModel, centroids, vocabCache);

                        ObjectWriter writer = mapper.writerFor(Tweet.class).withAttribute("score", score);

                        message = writer.writeValueAsString(tweet);

                        return message;

                    } catch (JsonProcessingException e) {
                        e.printStackTrace();
                        return "";
                    }

                });
    }


}
