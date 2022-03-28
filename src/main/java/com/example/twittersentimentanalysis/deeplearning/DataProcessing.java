package com.example.twittersentimentanalysis.deeplearning;

import java.util.*;

public class DataProcessing {

    public static String stringProcess(String text) {

        String urlPattern = "^(https?|ftp|file)://[-a-zA-Z0-9+&@#/%?=~_|!:,.;]*[-a-zA-Z0-9+&@#/%=~_|]";
        String emailPattern = "^[A-Z0-9._%+-]+@[A-Z0-9.-]+\\\\.[A-Z]{2,6}$";
//        String characterFilter = "[^p{L}p{M}p{N}p{P}p{Z}p{Cf}p{Cs}s]";
        String emojis = "/([\\u2700-\\u27BF]|[\\uE000-\\uF8FF]|\\uD83C[\\uDC00-\\uDFFF]|\\uD83D[\\uDC00-\\uDFFF]|[\\u2011-\\u26FF]|\\uD83E[\\uDD10-\\uDDFF])/g";
        String hashTag = "B([a-zA-Z]+b)";
        String mentions = "/B@w+/g";
        String newline =  "/\n/g";

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
                .replaceAll("\\p{Blank}{2,}+", " ") // remove redundant blanks
                .toLowerCase(); // to lower case

        String[] splitted = text.split(" ");
        String result = "";

        LinkedHashSet<String> set = new LinkedHashSet<>(Arrays.asList(splitted));

        for (String s : set) {
            result += s + " ";
        }

        return result;

    }

}