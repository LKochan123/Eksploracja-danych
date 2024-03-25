package org.example;

import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;
import org.apache.spark.sql.*;

import java.io.IOException;
import java.util.List;

import static org.apache.spark.sql.functions.*;
import static org.apache.spark.sql.functions.col;

public class LoadTags {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("LoadMovies")
                .master("local")
                .getOrCreate();

        Dataset<Row> df = spark.read().format("csv").option("header", "true").load("data/tags.csv");
        Dataset<Row> users = spark.read().format("csv").option("header", "true").load("data/users.csv");

        Dataset<Row> userTags = joinUserTags(users, df, spark);
        Dataset<Row> emailTags = groupByEmailAndAggregateTags(userTags, spark);

        List<String> tagsList = emailTags.select("tags").as(Encoders.STRING()).collectAsList();

        for (String tag : tagsList) {
            System.out.println(tag);
        }

//        df = transformDataframe(df);
//        df = countTags(df);
//        plot_stats_ym(df, "Number of tags in subsequent months", "Tags");

//        System.out.println("Excerpt of the dataframe content:");
//        users.show(5);
//
//        System.out.println("Dataframe's schema:");
//        users.printSchema();
    }

    private static Dataset<Row> transformDataframe(Dataset<Row> df) {
        return df.withColumn("datetime", functions.from_unixtime(df.col("timestamp")))
                .withColumn("year", year(col("datetime")))
                .withColumn("month", month(col("datetime")))
                .withColumn("day", dayofmonth(col("datetime")));
    }

    private static Dataset<Row> countTags(Dataset<Row> df) {
        return df.groupBy("year", "month").count().orderBy("year", "month");
    }

    private static void plot_stats_ym(Dataset<Row> df, String title, String label) {
        List<Double> x = df.select(expr("year+(month-1)/12")).as(Encoders.DOUBLE()).collectAsList();
        List<Double> y = df.select("count").as(Encoders.DOUBLE()).collectAsList();

        Plot plt = Plot.create();
        plt.plot().add(x, y).linestyle("-").label(label);
        plt.legend();
        plt.title(title);

        try {
            plt.show();
        } catch (IOException | PythonExecutionException e) {
            throw new RuntimeException(e);
        }
    }

    private static Dataset<Row> joinUserTags(Dataset<Row> users, Dataset<Row> tags, SparkSession spark) {
        users.createOrReplaceTempView("users");
        tags.createOrReplaceTempView("tags");

        String query = "SELECT * FROM users JOIN tags ON users.userId = tags.userId";
        return spark.sql(query);
    }

    private static Dataset<Row> groupByEmailAndAggregateTags(Dataset<Row> df, SparkSession spark) {
        df.createOrReplaceTempView("df");

        String query = "SELECT email, CONCAT_WS(' ', COLLECT_LIST(tag)) as tags " +
                "FROM df " +
                "GROUP BY email";

        return spark.sql(query);
    }
}
