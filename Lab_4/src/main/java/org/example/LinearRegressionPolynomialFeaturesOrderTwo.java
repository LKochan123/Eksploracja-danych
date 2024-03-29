package org.example;

import com.github.sh0nk.matplotlib4j.NumpyUtils;
import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.PolynomialExpansion;
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

import static org.apache.spark.sql.functions.col;


public class LinearRegressionPolynomialFeaturesOrderTwo {

    private static final String XY_002 = "xy-002";
    private static final String XY_003 = "xy-003";
    private static final String XY_004 = "xy-004";
    private static final String XY_005 = "xy-005";

    public static void main(String[] args) {
        SparkSession spark = createSparkSession();

//        Dataset<Row> df = processDf(spark, XY_002);
//
//        System.out.println("Proccessed dataset: " + XY_002);
//        df.show(5);

        processDataset(spark, XY_002, LinearRegressionPolynomialFeaturesOrderTwo::realFunction_002);
        processDataset(spark, XY_003, LinearRegressionPolynomialFeaturesOrderTwo::realFunction_002);
        processDataset(spark, XY_004, LinearRegressionPolynomialFeaturesOrderTwo::realFunction_004);
        processDataset(spark, XY_005, LinearRegressionPolynomialFeaturesOrderTwo::realFunction_005);
    }

    private static double realFunction_002(double x) {
        return -1.5 * x * x + 3 * x + 4;
    }

    private static double realFunction_004(double x) {
        return -10 * x * x + 500 * x - 25;
    }

    private static double realFunction_005(double x) {
        return (x + 4) * (x + 1) * (x - 3);
    }

    private static SparkSession createSparkSession() {
        return SparkSession.builder()
            .appName("VectorAssembler Example")
            .master("local")
            .getOrCreate();
    }

    private static Dataset<Row> loadFile(SparkSession spark, String path) {
        return spark.read()
            .format("csv")
            .option("header", "true")
            .option("inferSchema", "true")
            .load(path);
    }

    private static Dataset<Row> addPolynomialColumns(Dataset<Row> data) {
        data = data.withColumn("X2", col("X").multiply(col("X")));
        return data;
    }

    private static VectorAssembler configVector() {
        return new VectorAssembler()
            .setInputCols(new String[] {"X"})
            .setOutputCol("features");
    }

    private static PolynomialExpansion configPolynomial(int degree) {
        return new PolynomialExpansion()
            .setInputCol("features")
            .setOutputCol("polyFeatures")
            .setDegree(degree);
    }

    private static PipelineModel createPipeline(VectorAssembler v, PolynomialExpansion p, LinearRegression lr, Dataset<Row> df) {
        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] {v, p, lr});
        return pipeline.fit(df);
    }

    private static LinearRegression createLinearRegressionModel(int iter, double reg, double elasticNet) {
        return new LinearRegression()
            .setMaxIter(iter)
            .setRegParam(reg)
            .setElasticNetParam(elasticNet)
            .setFeaturesCol("features")
            .setLabelCol("Y");
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
        return fx.stream().map(xi -> lrModel.predict(Vectors.dense(xi, Math.pow(xi, 2)))).collect(Collectors.toList());
    }

    private static Dataset<Row> processDf(SparkSession spark, String file) {
        Dataset<Row> data = loadFile(spark, "data/" + file + ".csv");
        Dataset<Row> dataX2 = addPolynomialColumns(data);
        return configVector().transform(dataX2);
    }

    private static void processDataset(SparkSession spark, String file, Function<Double,Double> f_true) {
        Dataset<Row> df = processDf(spark, file);

        LinearRegression lr = createLinearRegressionModel(10, 0.3, 0.8);
        LinearRegressionModel lrModel = lr.fit(df);

//        printCoefficientsAndIntercept(lrModel);
        trainingSummary(lrModel.summary());

        List<Double> x = df.select("X").as(Encoders.DOUBLE()).collectAsList();
        List<Double> y = df.select("Y").as(Encoders.DOUBLE()).collectAsList();
        plot(x, y, lrModel, "Linear regression: " + file, f_true);
    }
}
