package com.example.twittersentimentanalysis.deeplearning;

import org.deeplearning4j.clustering.algorithm.Distance;
import org.deeplearning4j.clustering.cluster.Cluster;
import org.deeplearning4j.clustering.cluster.ClusterSet;
import org.deeplearning4j.clustering.cluster.Point;
import org.deeplearning4j.clustering.cluster.PointClassification;
import org.deeplearning4j.clustering.kmeans.KMeansClustering;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.nd4j.common.util.SerializationUtils;
import org.nd4j.linalg.api.ndarray.INDArray;

import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.*;

public class Clustering {

    private static final Logger logger = LoggerFactory.getLogger(Clustering.class);
    private static final long SEED = 239;
    private static final double MINIMUM_DISTANCE = 0.001;

    private static int nCentroid;
    private static List<Point> pointsList;

    public static List<Point> kMeansPlusPlus() {

        List<Point> remainingPoints = pointsList;

        // 1. Find and set the initial centroid
        Random rand = new Random(SEED);
        int randIndex = rand.nextInt(remainingPoints.size());

        logger.info("Generating centroid - 1");
        List<Point> centroids = new ArrayList<>();
        centroids.add(remainingPoints.get(randIndex));
        remainingPoints.remove(randIndex);

        // 2. To find each of the centroid
        for (int i = 1; i < nCentroid; i++) {

            logger.info("Generating centroid - " + (i + 1));

            // Collection of distances of points to the nearest centroid
            List<Double> nearestDistances = new ArrayList<>();

            // 2.5 For each point
            for (Point point : pointsList) {

                List<Double> distances = new ArrayList<>();

                // 2.6 Find the nearest centroid
                for (Point centroid : centroids) {

                    double currDistance = Transforms.allCosineDistances(centroid.getArray(), point.getArray()).getDouble(1, 1);

                    distances.add(currDistance);

                }

                // 2.6 Record its distance to the nearest centroid
                nearestDistances.add(Collections.min(distances));

            }

            // Collection of probability of points to be chosen as centroid
            List<Double> probabilities = new ArrayList<>();

            // 3. Find the point with the highest chance to be selected as centroid
            for (Double distance : nearestDistances) {

                // currDistance ^ 2 / Σ(distances ^ 2)
                double probability = Math.pow(distance, 2) / nearestDistances.stream().mapToDouble(curr -> Math.pow(curr, 2)).sum();

                probabilities.add(probability);

            }

            // 3.1 Select the point with the highest probability, set as next centroid
            int candidate = probabilities.indexOf(Collections.max(probabilities));
            centroids.add(remainingPoints.get(candidate));
            remainingPoints.remove(candidate);

        }

        return centroids;
    }

    // Training
    public static void train(Word2Vec w2vModel, int nCentroid, int minIteration) {

        Clustering.nCentroid = nCentroid;

        List<INDArray> wordVectors = new ArrayList<>();

        for (String word : w2vModel.vocab().words()) {
            wordVectors.add(w2vModel.getWordVectorMatrix(word));
        }

        logger.info(wordVectors.size() + " vectors extracted to create Point list");
        pointsList = Point.toPoints(wordVectors);
        logger.info(pointsList.size() + " Points created out of " + wordVectors.size() + " vectors");

        logger.info("Centroids initialization with K-means++");
        // 1. Initialization of centroids
        List<Point> centroids = kMeansPlusPlus();

        logger.info("Start Clustering " + pointsList.size() + " points/docs");

        int flagIndex = 0, index = 0;

        // 2. Loop through the data points
        while (flagIndex < nCentroid && (index + 1) < minIteration) {

            logger.info("Iteration: " + (index + 1));

            List<List<INDArray>> distancesToCenters = new ArrayList<>(2);
            distancesToCenters.add(new ArrayList<INDArray>());
            distancesToCenters.add(new ArrayList<INDArray>());

            // 2.1 Find the closest centroid and group up the data points
            for (Point point : pointsList) {
                INDArray d1 = Transforms.allCosineSimilarities(centroids.get(0).getArray(), point.getArray());
                INDArray d2 = Transforms.allCosineSimilarities(centroids.get(1).getArray(), point.getArray());

                if (d1.getDouble(0, 0) > d2.getDouble(0, 0)) {
                    distancesToCenters.get(0).add(point.getArray());
                } else if (d1.getDouble(0, 0) < d2.getDouble(0, 0)) {
                    distancesToCenters.get(1).add(point.getArray());
                }

            }

            // 2.2 Find the means of vectors, set the new centroids
            for (int i = 0; i < nCentroid; i++) {

                INDArray average = Nd4j.zeros(1, distancesToCenters.get(0).get(0).columns());
//                INDArray average = Nd4j.averageAndPropagate(distancesToCenters.get(i));

                for (INDArray distanceToCenter : distancesToCenters.get(i)) {
                    average.addi(distanceToCenter);
                }

                average.divi(pointsList.size());

//                System.out.println(average);
                System.out.println(centroids.get(i).getArray().sub(average).sum().getDouble(0, 0));

                if (Math.abs(centroids.get(i).getArray().sub(average).sum().getDouble(0, 0)) > MINIMUM_DISTANCE) {
                    centroids.get(i).setArray(average);
                } else {
                    flagIndex++;
                    System.out.println("++");
                }

            }

            index++;
        }

        // Saving found centroids vectors
        SerializationUtils.saveObject(centroids, new File("saved assets/cluster.dat"));

    }

    // ALTERNATIVE FROM DL4J, LOOKS NICE DOESN'T WORK, AFAIK
    public static void clusterTrain2(Word2Vec w2v, int nCentroid, int iterations) {
//1. create a kmeanscluster instance
        KMeansClustering kmc = KMeansClustering.setup(nCentroid, iterations, Distance.COSINE_DISTANCE, true);
        //2. iterate over rows in the paragraphvector and create a List of paragraph vectors
        int i = 0;
        List<INDArray> vectors = new ArrayList<INDArray>();
        for (String word : w2v.vocab().words()) {

            vectors.add(w2v.getWordVectorMatrix(word));

            if (i == 100)
                break;

            i++;
        }


        logger.info(vectors.size() + " vectors extracted to create Point list");
        List<Point> pointsLst = Point.toPoints(vectors);
        logger.info(pointsLst.size() + " Points created out of " + vectors.size() + " vectors");

        logger.info("Start Clustering " + pointsLst.size() + " points/docs");
        ClusterSet cs = kmc.applyTo(pointsLst);
        vectors = null;
        pointsLst = null;

        logger.info("Finish  Clustering");

        List<Cluster> clsterLst = cs.getClusters();

        System.out.println("\nCluster Centers:");
        for (Cluster c : clsterLst) {
            Point center = c.getCenter();
            System.out.println(center.getArray());
        }

        Point newpoint = Point.toPoints(w2v.getWordVectorMatrix("sedih")).get(0);
        PointClassification pc = cs.classifyPoint(newpoint);
        System.out.println(pc.getCluster().getCenter().getId());

    }

    // Testing
    public static double test(Word2Vec w2vModel, List<Point> centroids, String word) {

        logger.info("Trying to classify a point that was used for generating the Clusters");
        Point point = new Point(w2vModel.getWordVectorMatrix(word));

        if (point.getArray() == null)
            return 0.0;

        INDArray d1 = Transforms.allCosineSimilarities(centroids.get(0).getArray(), point.getArray());
        INDArray d2 = Transforms.allCosineSimilarities(centroids.get(1).getArray(), point.getArray());

        if (d1.getDouble(0, 0) > d2.getDouble(0, 0)) {
            return -1.0 * d2.getDouble(1, 1);
        } else {
            return 1.0 * d1.getDouble(1,1);
        }
    }


}

