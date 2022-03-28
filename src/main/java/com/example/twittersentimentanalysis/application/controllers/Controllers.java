package com.example.twittersentimentanalysis.application.controllers;

import com.example.twittersentimentanalysis.application.Tweet;
import com.example.twittersentimentanalysis.deeplearning.DataProcessing;
import com.example.twittersentimentanalysis.deeplearning.SentimentAnalysis;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectWriter;
import lombok.SneakyThrows;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Flux;

import javax.naming.ldap.Control;
import java.io.*;

@RequestMapping("/tweets")
@RestController
public class Controllers {

    // Training Materials
    private final Logger logger = LoggerFactory.getLogger(Control.class);
    private final Word2Vec w2vModel;
    private final ComputationGraph classifier;

    @Autowired
    private WebClient webClient;

    @SneakyThrows
    public Controllers() {
        // files reading
        logger.info("Reading files ...");

        w2vModel = WordVectorSerializer.loadFullModel("saved assets/w2v.vec");
        classifier = ComputationGraph.load(new File("saved assets/classification_model"), true);

        logger.info("End of files reading");

    }

    @GetMapping(value = "/getTweets", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<String> sendingTweets() {

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
                        String text = DataProcessing.stringProcess(tweet.getText());

                        double score_classification = SentimentAnalysis.sentimentAnalysis(w2vModel, classifier, text);

                        ObjectWriter writer = mapper.writerFor(Tweet.class).withAttribute("score", score_classification);

                        message = writer.writeValueAsString(tweet);

                        return message;

                    } catch (IOException e) {
                        return e.getMessage();
                    }

                });
    }


}
