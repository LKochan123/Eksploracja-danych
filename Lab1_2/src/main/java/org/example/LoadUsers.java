package org.example;

import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.IOException;
import java.util.List;

import static org.apache.spark.sql.functions.*;

public class LoadUsers {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("LoadUsers")
                .master("local")
                .getOrCreate();

        System.out.println("Using Apache Spark v" + spark.version());

        Dataset<Row> users = spark.read().format("csv").option("header", "true").load("data/users.csv");
        Dataset<Row> ratings = spark.read().format("csv").option("header", "true").load("data/ratings.csv");

        Dataset<Row> df_ur = joinUsersRatings(users, ratings, spark);
        Dataset<Row> df_ur_stats = groupByEmailAndCountStats(df_ur);

        plot_stats(df_ur_stats, "Number of ratings vs avg ratings");

//        df_ur_stats.show();

//        System.out.println("Excerpt of the dataframe content:");
//        df.show(20);
//        System.out.println("Dataframe's schema:");
//        df.printSchema();
    }

    private static Dataset<Row> joinUsersRatings(Dataset<Row> users, Dataset<Row> ratings, SparkSession spark) {
        users.createOrReplaceTempView("users");
        ratings.createOrReplaceTempView("ratings");
        
        String query = "SELECT * FROM users JOIN ratings ON users.userId = ratings.userId";
        return spark.sql(query);
    }

    private static Dataset<Row> groupByEmailAndCountStats(Dataset<Row> df_ur) {
        return df_ur.groupBy(col("email")).agg(
                avg("rating"),
                count("rating")
        ).orderBy(col("avg(rating)").desc());
    }

    private static void plot_stats(Dataset<Row> df, String title) {
        List<Double> x = df.select("avg(rating)").as(Encoders.DOUBLE()).collectAsList();
        List<Double> y = df.select("count(rating)").as(Encoders.DOUBLE()).collectAsList();

        Plot plt = Plot.create();
        plt.plot().add(x, y, "o");
        plt.legend();
        plt.title(title);

        try {
            plt.show();
        } catch (IOException | PythonExecutionException e) {
            throw new RuntimeException(e);
        }
    }
}
