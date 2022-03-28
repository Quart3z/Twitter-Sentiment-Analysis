package com.example.twittersentimentanalysis.application.configurations;

import com.example.twittersentimentanalysis.application.Tweet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Sinks;


@Configuration
public class Configurations {

    private static final String TOKEN = "AAAAAAAAAAAAAAAAAAAAANHOaAEAAAAALnE000uieeI%2FkKhsXMKJDhiykps%3DxDBjMVNF8Gzv0NEx1KFd15bOvHBagIibcD5BKQIrLcK1ypKgJ7";
    private static final String ENDPOINT = "https://api.twitter.com/2/tweets/search/stream?expansions=author_id";
    private final Logger logger = LoggerFactory.getLogger(Configurations.class);

    // requests sending
    @Bean
    public WebClient webClient() {

        logger.info("Config");

        return WebClient.builder()
                .defaultHeaders(httpHeaders -> {
                    httpHeaders.set("Authorization", "Bearer " + TOKEN);
                    httpHeaders.set("Retry-After", "300000");
                })
                .baseUrl(ENDPOINT)
                .build();

    }

    @Bean
    public Sinks.Many<Tweet> sink() {
        return Sinks.many().replay().latest();
    }

    @Bean
    public Flux<Tweet> flux(Sinks.Many<Tweet> sink) {
        return sink.asFlux().cache();
    }

}


