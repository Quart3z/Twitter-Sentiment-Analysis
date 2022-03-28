package com.example.twittersentimentanalysis.application;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.annotation.JsonAppend;

import java.io.Serializable;
import java.util.Map;


@JsonAppend(attrs = {
        @JsonAppend.Attr(value = "score")
})
@JsonIgnoreProperties({"includes", "matching_rules"})
public class Tweet implements Serializable {

    private String text;
    private String id;

    @JsonProperty("data")
    private void unpackNested(Map<String, Object> data) {
        this.text = (String) data.get("text");
        this.id = (String) data.get("id");
    }

    public String getText() {
        return text;
    }

    public String getId() {
        return id;
    }

}

class Data {

    @JsonProperty("author_id")
    private String authorId;

    @JsonProperty("id")
    private String id;


    private String text;

    public String getText() {
        return text;
    }

    @JsonProperty("text")
    public void setText(String text) {
        this.text = text;
    }

}