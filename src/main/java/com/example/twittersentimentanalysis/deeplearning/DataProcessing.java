package com.example.twittersentimentanalysis.deeplearning;

import java.util.*;

public class DataProcessing {

    public static String stringProcess(String text) {

        String urlPattern = "https?://\\S+\\s?";
        String emailPattern = "^[A-Z0-9._%+-]+@[A-Z0-9.-]+\\\\.[A-Z]{2,6}$";
//        String characterFilter = "[^p{L}p{M}p{N}p{P}p{Z}p{Cf}p{Cs}s]";
        String emojis = "/([\\u2700-\\u27BF]|[\\uE000-\\uF8FF]|\\uD83C[\\uDC00-\\uDFFF]|\\uD83D[\\uDC00-\\uDFFF]|[\\u2011-\\u26FF]|\\uD83E[\\uDD10-\\uDDFF])/g";
        String hashTag = "#";
        String mentions = "@\\s*(\\w*)";
        String newline = "/\n/g";

        text = text
                .replaceAll(newline, " ")
                .replaceAll(urlPattern, "")
//                .replaceAll(characterFilter, "")
                .replaceAll(emailPattern, "")  // remove email
                .replaceAll(hashTag, "") // remove hashtag
                .replaceAll(mentions, "")
                .replaceAll(emojis, "")
                .replaceAll("\\p{Punct}", "")  // remove punctuation
                .replaceAll("\\d", "") // remove digit
                .trim();

        return text;

    }

}
