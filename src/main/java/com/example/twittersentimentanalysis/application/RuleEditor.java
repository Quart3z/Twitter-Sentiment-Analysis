package com.example.twittersentimentanalysis.application;

import org.apache.http.HttpResponse;
import org.apache.http.client.HttpClient;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.ContentType;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.HttpClientBuilder;

import java.io.IOException;
import java.util.Scanner;

public class RuleEditor {

    public static void main(String[] args) throws IOException, InterruptedException {

        final String key = "REakpNScSciahRi1VzBZTod7r";
        final String secret = "DiR0OLO0wP8tsHtR7voFOWg3Ft8kwkFmlW6nBTK1r9JinpHZGU";
        final String token = "AAAAAAAAAAAAAAAAAAAAANHOaAEAAAAALnE000uieeI%2FkKhsXMKJDhiykps%3DxDBjMVNF8Gzv0NEx1KFd15bOvHBagIibcD5BKQIrLcK1ypKgJ7";

//        final String endpoint = "https://api.twitter.com/2/users/2989267819/following";
        final String endpoint = "https://api.twitter.com/2/tweets/search/stream/rules";

        //bounding_box:[100.085756871 0.773131415201 119.181903925 6.92805288332]
        final String body = "{\n" +
                "  \"add\": [\n" +
                "    {\"value\": \"najib -is:retweet -has:mentions -has:media -has:images -has:videos -has:links \"} \n" +
                "  ]\n" +
                "}";

        // 1508292450443730944
//        final String body = "{\"delete\":  \n" +
//                "   {\"ids\": \n" +
//                "       [\"1506857104136876033\" ]\n" +
//                "   }\n" +
//                "}";

        HttpClient httpClient = HttpClientBuilder.create().build();
        HttpPost request = new HttpPost(endpoint);
//        HttpGet request = new HttpGet(endpoint);

        StringEntity entity = new StringEntity(body, ContentType.APPLICATION_JSON);
        request.setHeader("Authorization", "Bearer " + token);
        request.setEntity(entity);

        try {
            HttpResponse response = httpClient.execute(request);
            Scanner scanner = new Scanner(response.getEntity().getContent());

            while (scanner.hasNext()) {
                System.out.println(scanner.nextLine());
            }

        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}
