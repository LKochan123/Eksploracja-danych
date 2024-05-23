package org.example;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class PiComputeApp implements Serializable {
    private static long counter = 0;

    public static void main(String[] args) {
        int slices = 100;
        int n = 1_000_000 * slices;

        System.out.printf("Table of %d elements will be divided into %d partitions.\n" +
                "Each partition will be processed as separate task\n", n, slices);

        long t0 = System.currentTimeMillis();

        SparkSession spark = SparkSession.builder()
            .appName("LoadDatasetDistributed")
            .master("spark://172.22.0.2:7077")
            .getOrCreate();

        Dataset<Row> df = spark.read().format("csv")
            .option("header", "true")
            .option("inferschema","true")
            .load("data/owid-energy-data.csv.gz");

        df.show(5);
        df = df.select("country","year", "population", "electricity_demand").where("country like \'Po%\' AND year >= 2000");
        df.show(5);
        df.printSchema();

        SparkContext sparkContext = spark.sparkContext();
        JavaSparkContext ctx = new JavaSparkContext(sparkContext);

        long t1 = System.currentTimeMillis();
        System.out.println("Session initialized in " + (t1 - t0) + " ms");

        List<Integer> l = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            l.add(i);
        }

        JavaRDD<Integer> dataSet = ctx.parallelize(l, slices);
        long t2 = System.currentTimeMillis();
        System.out.println("Initial dataframe built in " + (t2 - t1) + " ms");

        final int count = dataSet.map(integer -> {
            double x = Math.random() * 2 - 1;
            double y = Math.random() * 2 - 1;
            return (x * x + y * y < 1) ? 1 : 0;
        }).reduce((a, b) -> a + b);

        long t3 = System.currentTimeMillis();
        System.out.println("Map-reduce time " + (t3 - t2) + " ms");
        System.out.println("Pi is roughly " + 4.0 * count / n);
        spark.stop();
    }
}
