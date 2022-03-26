package com.example.tweetsclassifier.application;

import com.fasterxml.jackson.annotation.JsonAnySetter;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonSetter;
import com.fasterxml.jackson.databind.ObjectWriter;
import com.fasterxml.jackson.databind.annotation.JsonAppend;

import java.io.Serializable;


@JsonAppend(attrs = {
        @JsonAppend.Attr(value = "score")
})
@JsonIgnoreProperties({"includes", "matching_rules"})
public class Tweet implements Serializable {

    @JsonProperty("data")
    private  Data data;

    public String getText(){
        return  data.getText();
    }

}

class Data{

    @JsonProperty("author_id")
    private  String authorId;

    @JsonProperty("id")
    private String id;

    @JsonProperty("text")
    private String text;

    public String getText(){
        return text;
    }

}