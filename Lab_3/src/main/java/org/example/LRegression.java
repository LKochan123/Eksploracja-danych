package org.example;

import com.github.sh0nk.matplotlib4j.NumpyUtils;
import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;


public class LRegression {

    private static final String FILE = "xy-010";

    public static void main(String[] args) {
        SparkSession spark = createSparkSession();
        Dataset<Row> data = loadFile(spark, "data/" + FILE + ".csv");
        Dataset<Row> dfTransformed = configVector().transform(data);

        LinearRegression lr = createLinearRegressionModel(10, 0.3, 0.8);
        LinearRegressionModel lrModel = lr.fit(dfTransformed);
//        printCoefficientsAndIntercept(lrModel);
        trainingSummary(lrModel.summary());

//        List<Double> valuesList = convertArrayDoubleToList(lrModel.summary().objectiveHistory());
//        plotObjectiveHistory(valuesList);

        List<Double> x = dfTransformed.select("X").as(Encoders.DOUBLE()).collectAsList();
        List<Double> y = dfTransformed.select("Y").as(Encoders.DOUBLE()).collectAsList();
        plot(x, y, lrModel, "Linear regression", null);
    }

    private static SparkSession createSparkSession() {
        return SparkSession.builder().appName("VectorAssembler Example").master("local").getOrCreate();
    }

    private static Dataset<Row> loadFile(SparkSession spark, String path) {
        return spark.read().format("csv").option("header", "true").option("inferSchema", "true").load(path);
    }

    //    You need to change this function to each file (according to the instructions)
    private static double realFunction(double x) {
        return (x + 4) * (x + 1) * (x - 3);
    }

    private static VectorAssembler configVector() {
        return new VectorAssembler().setInputCols(new String[] {"X"}).setOutputCol("features");
    }

    private static LinearRegression createLinearRegressionModel(int iteration, double reg, double elasticNet) {
        return new LinearRegression().setMaxIter(iteration).setRegParam(reg).setElasticNetParam(elasticNet).setFeaturesCol("features").setLabelCol("Y");
    }

    private static void printCoefficientsAndIntercept(LinearRegressionModel lrModel) {
        System.out.println("Coefficients: " + lrModel.coefficients());
        System.out.println("Intercept: " + lrModel.intercept());
    }

    private static void trainingSummary(LinearRegressionTrainingSummary trainingSummary) {
        System.out.println("numIterations: " + trainingSummary.totalIterations());
        System.out.println("objectiveHistory: " + Vectors.dense(trainingSummary.objectiveHistory()));
        trainingSummary.residuals().show(100);
        System.out.println("MSE: " + trainingSummary.meanSquaredError());
        System.out.println("RMSE: " + trainingSummary.rootMeanSquaredError());
        System.out.println("MAE: " + trainingSummary.meanAbsoluteError());
        System.out.println("r2: " + trainingSummary.r2());
    }

    private static void plotObjectiveHistory(List<Double> lossHistory) {
        List<Double> x = IntStream.range(0, lossHistory.size()).mapToDouble(d -> d).boxed().toList();
        Plot plt = Plot.create();
        plt.plot().add(x, lossHistory).label("loss");
        plt.xlabel("Iteration");
        plt.ylabel("Loss");
        plt.title("Loss history");
        plt.legend();

        try {
            plt.show();
        } catch (IOException | PythonExecutionException e) {
            throw new RuntimeException(e);
        }
    }

    private static List<Double> convertArrayDoubleToList(double[] values) {
        return Arrays.stream(values).boxed().collect(Collectors.toList());
    }

    private static void plot(List<Double> x, List<Double> y, LinearRegressionModel lrModel, String title, Function<Double, Double> fTrue) {

        Plot plt = Plot.create();
        plt.plot().add(x, y, "o").label("data");
        plt.title(title);
        plt.xlabel("X");
        plt.ylabel("Y");

        double xmin = x.stream().min(Double::compare).get();
        double xmax = x.stream().max(Double::compare).get();
        double xdelta = 0.05 * (xmax - xmin);

        List<Double> fx = NumpyUtils.linspace(xmin - xdelta, xmax + xdelta, 100);
        List<Double> predictions = predictions(fx, lrModel);
        plt.plot().add(fx, predictions).label("Regression line");

        if (fTrue != null) {
            List<Double> fTrueY = fx.stream().map(fTrue).collect(Collectors.toList());
            plt.plot().add(fx, fTrueY).label("True function");
        }

        plt.legend();

        try {
            plt.show();
        } catch (IOException | PythonExecutionException e) {
            throw new RuntimeException(e);
        }
    }

    private static List<Double> predictions(List<Double> fx, LinearRegressionModel lrModel) {
        return fx.stream().map(xi -> lrModel.predict(Vectors.dense(xi))).collect(Collectors.toList());
    }
}
