package org.example;

import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;
import org.apache.spark.sql.*;

import java.io.IOException;
import java.util.List;

import static org.apache.spark.sql.functions.*;

public class LoadRatings {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder().appName("LoadMovies").master("local").getOrCreate();

        Dataset<Row> df = spark.read().format("csv").option("header", "true").load("data/ratings.csv");

        df = transformDataframe(df);
        df = countRatings(df);
        plot_stats_ym(df, "Number of ratings in subsequent months", "rating");

        System.out.println("Excerpt of the dataframe content:");
        df.show(5);

        System.out.println("Dataframe's schema:");
        df.printSchema();

    }

    private static Dataset<Row> transformDataframe(Dataset<Row> df) {
        return df.withColumn("datetime", functions.from_unixtime(df.col("timestamp"))).withColumn("year", year(col("datetime"))).withColumn("month", month(col("datetime"))).withColumn("day", dayofmonth(col("datetime")));
    }

    private static Dataset<Row> countRatings(Dataset<Row> df) {
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
}
