package org.example;

import com.github.sh0nk.matplotlib4j.NumpyUtils;
import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.PolynomialExpansion;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.StructType;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;


public class LinearRegressionPolynomialFeaturesOrderTwo {

    private static final String XY_002 = "xy-002";
    private static final String XY_003 = "xy-003";
    private static final String XY_004 = "xy-004";
    private static final String XY_005 = "xy-005";
    private static final int degree = 2;

    public static void main(String[] args) {
        SparkSession spark = createSparkSession();

//        processDataset(spark, XY_002, degree, LinearRegressionPolynomialFeaturesOrderTwo::realFunction_002);
//        processDataset(spark, XY_003, degree, LinearRegressionPolynomialFeaturesOrderTwo::realFunction_002);
//        processDataset(spark, XY_004, degree, LinearRegressionPolynomialFeaturesOrderTwo::realFunction_004);
        processDataset(spark, XY_005, degree, LinearRegressionPolynomialFeaturesOrderTwo::realFunction_005);
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

    private static void plot(SparkSession spark, List<Double> x, List<Double> y, PipelineModel pipelineModel, String title, Function<Double, Double> fTrue) {

        Plot plt = Plot.create();
        plt.plot().add(x, y, "o").label("data");
        plt.title(title);
        plt.xlabel("X");
        plt.ylabel("Y");

        double xmin = x.stream().min(Double::compare).get();
        double xmax = x.stream().max(Double::compare).get();
        double xdelta = 0.05 * (xmax - xmin);

        List<Double> fx = NumpyUtils.linspace(xmin - xdelta, xmax + xdelta, 100);
        List<Row> rows = predictions(fx);

        StructType schema = new StructType().add("X", "double");
        Dataset<Row> df_test = spark.createDataFrame(rows, schema);
        Dataset<Row> df_pred = pipelineModel.transform(df_test);

        List<Double> predictions = df_pred.select("prediction").as(Encoders.DOUBLE()).collectAsList();
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

    private static List<Row> predictions(List<Double> fx) {
        return fx.stream().map(RowFactory::create).collect(Collectors.toList());
    }

    private static List<Dataset<Row>> trainTestSplit(Dataset<Row> df) {
        long rowsCount = df.count();
        int trainCount = (int) (rowsCount * 0.7);

        Dataset<Row> df_train = df.select("*").limit(trainCount);
        Dataset<Row> df_test = df.select("*").offset(trainCount);

        return Arrays.asList(df_train, df_test);
    }

    private static List<Dataset<Row>> randomTrainTestSplit(Dataset<Row> df) {
        df = df.orderBy(org.apache.spark.sql.functions.rand(3));
        Dataset<Row>[] dfs = df.randomSplit(new double[] {0.7, 0.3});
        Dataset<Row> df_train = dfs[0];
        Dataset<Row> df_test = dfs[1];
        return Arrays.asList(df_train, df_test);
    }

    private static void pipelineModelMetrics(PipelineModel model, Dataset<Row> df_test) {
        Dataset<Row> df_test_prediction = model.transform(df_test);
        RegressionEvaluator evaluator = new RegressionEvaluator()
                .setLabelCol("Y")
                .setPredictionCol("prediction")
                .setMetricName("rmse");

        double rmse = evaluator.evaluate(df_test_prediction);
        evaluator.setMetricName("r2");
        double r2 = evaluator.evaluate(df_test_prediction);

        System.out.println("Rmse: " + rmse);
        System.out.println("R2: " + r2);
    }


    private static void processDataset(SparkSession spark, String filename, int degree, Function<Double,Double> f_true) {
        Dataset<Row> df = loadFile(spark, "data/" + filename + ".csv");

        VectorAssembler vectorAssembler = configVector();
        PolynomialExpansion polynomialExpansion = configPolynomial(degree);
        LinearRegression lr = createLinearRegressionModel(10, 0.3, 0.8)
                .setFeaturesCol("polyFeatures");


        List<Dataset<Row>> arr = randomTrainTestSplit(df);
        Dataset<Row> df_train = arr.get(0);
        Dataset<Row> df_test = arr.get(1);

        PipelineModel pipelineModel = createPipeline(vectorAssembler, polynomialExpansion, lr, df_train);
        pipelineModelMetrics(pipelineModel, df_test);

        List<Double> x = df_test.select("X").as(Encoders.DOUBLE()).collectAsList();
        List<Double> y = df_test.select("Y").as(Encoders.DOUBLE()).collectAsList();
        plot(spark, x, y, pipelineModel, "Linear regression: " + filename, f_true);
    }
}
