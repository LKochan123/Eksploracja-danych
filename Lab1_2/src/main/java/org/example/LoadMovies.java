package org.example;

import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;
import org.apache.spark.sql.*;

import java.io.IOException;
import java.util.List;

import static org.apache.spark.sql.functions.*;

public class LoadMovies {

    private static final String CSV_FORMAT = "csv";
    private static final String MOVIES_FILE_PATH = "data/movies.csv";
    private static final String TITLE_PATTERN = "^(.*?)\\s*\\((\\d{4})\\)\\s*$";

    public static void main(String[] args) {
        SparkSession spark = createSparkSession();
        Dataset<Row> df = readDataframe(spark, MOVIES_FILE_PATH);
        Dataset<Row> df2 = addCurrentDate(df);
        Dataset<Row> df_transformed = transformDataframe(df2);
        Dataset<Row> df_exploded = explodeDataframe(df_transformed);

        //        showDistinctGenres(df_exploded);
        //        List<String> genreArr = collectDistinctGenres(df_exploded);
        //        Dataset<Row> df_multigenre = createMultiGenreDataFrame(df_transformed, genreArr);

        Dataset<Row> df_ratings = readDataframe(spark, "data/ratings.csv");
        
        Dataset<Row> df_mr = joinMoviesAndRatings(df_ratings, df_transformed);
        Dataset<Row> df_grouped = groupDataset(df_mr);

        Dataset<Row> df_release = releaseToRatingYear(df_mr);
        Dataset<Row> filtered = df_release.filter("release_to_rating_year IS NOT NULL");
        Dataset<Row> df_release_grouped = groupByReleaseCounting(filtered);

        Dataset<Row> tf_dr = transformAndJoinWithRatings(df_release);
        Dataset<Row> dfStats = calcStatsByGenre(tf_dr);

        filterAndDisplayGenresWithHighAverageRatings(tf_dr, df_ratings, spark);

//        displayTopNGenres(dfStats, 3);

//        List<Double> releaseToRatingYears = df_release_grouped.select(col("release_to_rating_year"))
//                .as(Encoders.DOUBLE())
//                .collectAsList();
//
//        List<Double> counts = df_release_grouped.select(col("count"))
//                .as(Encoders.DOUBLE())
//                .collectAsList();

//        plot_histogram_v2(releaseToRatingYears, counts, "Rozkład różnicy lat pomiędzy oceną a wydaniem filmu");

//        Dataset<Row> sampled = df_release.sample(false, 0.0001);
//
//        List<Double> yearDifferences = sampled.select(col("release_to_rating_year"))
//                .as(Encoders.DOUBLE())
//                .collectAsList();

//        List<Double> X1 = averageRatings(df_grouped);
//        List<Double> X2 = ratingsCount(df_grouped);
//        List<Double> X3 = moreComplexGrouping(df_grouped);

//        plot_histogram(X3, "Rozkład l. ocen dla avg_rating >= 3.5. i rating_cnt > 20.");
//        plot_histogram(yearDifferences, "Rozkład różnicy lat pomiędzy oceną a wydaniem filmu");

//        System.out.println("Excerpt of the dataframe content:");
//        dfStats.show(10);
//
//        System.out.println("Dataframe's schema:");
//        dfStats.printSchema();
    }

    private static SparkSession createSparkSession() {
        return SparkSession.builder().appName("LoadMovies").master("local").getOrCreate();
    }

    private static Dataset<Row> readDataframe(SparkSession spark, String path) {
        return spark.read().format(CSV_FORMAT).option("header", "true").load(path);
    }

    private static Dataset<Row> addCurrentDate(Dataset<Row> df) {
        return df.withColumn("rok", year(now())).withColumn("miesiac", month(now())).withColumn("dzien", day(now())).withColumn("godzina", hour(now()));
    }

    private static Dataset<Row> transformDataframe(Dataset<Row> df) {
        return df.withColumn("title2", when(regexp_extract(col("title"), TITLE_PATTERN, 1).equalTo(""), col("title")).otherwise(regexp_extract(col("title"), TITLE_PATTERN, 1))).withColumn("year", regexp_extract(col("title"), TITLE_PATTERN, 2)).withColumn("genres_array", split(col("genres"), "\\|")).drop("title").withColumnRenamed("title2", "title");
    }

    private static Dataset<Row> explodeDataframe(Dataset<Row> df) {
        return df.withColumn("genre", explode(col("genres_array"))).drop("genres_array");
    }

    private static void showDistinctGenres(Dataset<Row> df) {
        df.select("genre").distinct().show(false);
    }

    private static List<String> collectDistinctGenres(Dataset<Row> df) {
        return df.select("genre").distinct().as(Encoders.STRING()).collectAsList();
    }

    private static Dataset<Row> createMultiGenreDataFrame(Dataset<Row> df, List<String> genreArr) {
        Dataset<Row> df_multigenre = df;
        for (String s : genreArr) {
            if (s.equals("(no genres listed)")) continue;
            df_multigenre = df_multigenre.withColumn(s, array_contains(col("genres_array"), s));
        }
        return df_multigenre;
    }

    private static Dataset<Row> joinMoviesAndRatings(Dataset<Row> ratings, Dataset<Row> movies) {
        return movies.join(ratings, "movieId", "inner");
    }

    private static Dataset<Row> groupDataset(Dataset<Row> df_mr) {
        return df_mr.groupBy("title").agg(min("rating").alias("min_rating"), avg("rating").alias("avg_rating"), max("rating").alias("max_rating"), count("rating").alias("rating_cnt")).orderBy(col("rating_cnt").desc());
    }

    private static List<Double> averageRatings(Dataset<Row> df) {
        return df.select("avg_rating").where("rating_cnt>=0").as(Encoders.DOUBLE()).collectAsList();
    }

    private static List<Double> ratingsCount(Dataset<Row> df) {
        return df.select("rating_cnt").where("avg_rating>=4.5").as(Encoders.DOUBLE()).collectAsList();
    }

    private static List<Double> moreComplexGrouping(Dataset<Row> df) {
        return df.select("rating_cnt").where("avg_rating >= 3.5 AND rating_cnt > 20").as(Encoders.DOUBLE()).collectAsList();
    }

    private static void plot_histogram(List<Double> x, String title) {
        Plot plt = Plot.create();
        plt.hist().add(x).bins(50);
        plt.title(title);

        try {
            plt.show();
        } catch (IOException | PythonExecutionException e) {
            throw new RuntimeException(e);
        }
    }

    private static void plot_histogram_v2(List<Double> x, List<Double> weights, String title) {
        Plot plt = Plot.create();
        plt.hist().add(x).weights(weights).bins(50);
        plt.title(title);

        try {
            plt.show();
        } catch (IOException | PythonExecutionException e) {
            throw new RuntimeException(e);
        }
    }

    private static Dataset<Row> releaseToRatingYear(Dataset<Row> df) {
        df = df.withColumn("datetime", functions.from_unixtime(df.col("timestamp")));
        df = df.withColumn("rating_year", functions.year(df.col("datetime")));
        df = df.withColumn("release_to_rating_year", df.col("rating_year").minus(df.col("year")));
        df = df.drop("timestamp").drop("miesiac").drop("dzien").drop("godzina").drop("genres_array").drop("rating_year");
        return df;
    }

    private static Dataset<Row> groupByReleaseCounting(Dataset<Row> df) {
        return df.groupBy(col("release_to_rating_year")).count().orderBy(col("release_to_rating_year"));
    }

    public static Dataset<Row> transformAndJoinWithRatings(Dataset<Row> movies) {
        return movies.withColumn("genre", explode(split(col("genres"), "\\|"))).drop("genres").drop("release_to_rating_year").drop("rok");
    }

    public static Dataset<Row> calcStatsByGenre(Dataset<Row> df) {
        return df.groupBy(col("genre")).agg(min("rating"), avg("rating"), max("rating"), count("rating"));
    }

    public static void displayTopNGenres(Dataset<Row> aggregatedDf, int n) {
        aggregatedDf.orderBy(col("avg(rating)").desc()).show(n);
        aggregatedDf.orderBy(col("count(rating)").desc()).show(n);
    }

    public static void filterAndDisplayGenresWithHighAverageRatings(Dataset<Row> mr, Dataset<Row> ratings, SparkSession spark) {
        mr.createOrReplaceTempView("movies_ratings");
        ratings.createOrReplaceTempView("ratings");

        String query = """
                SELECT genre, AVG(rating) AS avg_rating, COUNT(rating) 
                FROM movies_ratings GROUP BY genre 
                HAVING AVG(rating) > (SELECT AVG(rating) FROM ratings) 
                ORDER BY avg_rating DESC""";

        spark.sql(query).show();
    }
}
