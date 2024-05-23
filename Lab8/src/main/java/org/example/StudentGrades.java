package org.example;

import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;
import org.apache.spark.api.java.function.ForeachFunction;
import org.apache.spark.ml.classification.BinaryLogisticRegressionTrainingSummary;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.*;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataTypes;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.apache.spark.sql.functions.*;

public class StudentGrades {

    private static final String CSV_FORMAT = "csv";
    private static final String DATA_FILE_PATH = "src/main/resources/";
    private static final String[] FEATURE_NAMES = {"OcenaC", "OcenaCpp", "timestamp"};

    public static void main(String[] args) {
        // 1. Data loading and preprocessing
        SparkSession spark = createSparkSession();
        spark.udf().register("max_vector_element", new MaxVectorElement(), DataTypes.DoubleType);

        Dataset<Row> df = loadDataframe(spark, DATA_FILE_PATH + "egzamin-cpp.csv", ";");
        Dataset<Row> df_exam = prepareExamDataset(df);

        // 2. Logistic regression analysis
        VectorAssembler va = configVectorAssembler();
        Dataset<Row> df_exam_va = va.transform(df_exam);
        LogisticRegression lr = configureLogisticRegression(100, 0.1, 0);
        LogisticRegressionModel lrModel = buildLogisticRegressionModel(lr, df_exam_va);
        Dataset<Row> df_with_predictions = calcPredictions(lrModel, df_exam_va);
//        analyzePredictions(df_with_predictions.limit(1), lrModel);
//        printLogisticRegressionEquation(lrModel);
        Dataset<Row> df_with_prob = concatPredictionAndProbColumn(df_exam, df_with_predictions);
//        saveFile(df_with_prob, "exam-with-classification.csv");


        // 3. Logistic regression scores
        Object[] arr = trainAndTest(df_exam_va);
        LogisticRegressionModel model = (LogisticRegressionModel) arr[0];
        Dataset<Row> df_test = (Dataset<Row>) arr[1];

        BinaryLogisticRegressionTrainingSummary trainingSummary = calcSummary(model);
        double[] objectiveHistory = calcObjectiveHistory(trainingSummary);
//        plotObjectiveHistory(objectiveHistory);
        Dataset<Row> roc = trainingSummary.roc();
//        plotROC(roc);
//        showOtherMetrics(trainingSummary);
        Dataset<Row> dfFMeasures = trainingSummary.fMeasureByThreshold();
        double maxFMeasure = calcMaxFMeasure(dfFMeasures);
        double bestThreshold = findThresholdBaseOnMaxFMeasure(dfFMeasures, maxFMeasure);

        lrModel.setThreshold(bestThreshold);
        Dataset<Row> predictions = makePredictions(lrModel, df_test);
        MulticlassClassificationEvaluator eval = configEvaluator();
//        calcOtherMetricsBaseOnEvaluatorAndPredictions(eval, predictions);

        // 4. LogisticRegressionGrid
        addClassificationToGrid(spark, lrModel);

//        showDataframeContentAndSchema(dfFMeasures);
    }

    private static SparkSession createSparkSession() {
        return SparkSession.builder()
            .appName("LogisticRegressionOnExam")
            .master("local")
            .getOrCreate();
    }

    private static Dataset<Row> loadDataframe(SparkSession spark, String path, String delimeter) {
        return spark.read().format(CSV_FORMAT)
            .option("header", "true")
            .option("delimiter", delimeter)
            .load(path);
    }

    private static void showDataframeContentAndSchema(Dataset<Row> df) {
        System.out.println("Excerpt of the dataframe content:");
        df.show(10);

        System.out.println("Dataframe's schema:");
        df.printSchema();
    }

    private static Dataset<Row> prepareExamDataset(Dataset<Row> df) {
        df = df.withColumn("timestamp", unix_timestamp(col("DataC"), "yyyy-MM-dd"));
        df = df.withColumn("Wynik", when(col("Egzamin").geq(3.0), 1).otherwise(0));

        df = df
            .withColumn("OcenaC", col("OcenaC").cast("double"))
            .withColumn("OcenaCpp", col("OcenaCpp").cast("double"))
            .withColumn("Egzamin", col("Egzamin").cast("double"));

        return df;
    }

    private static Dataset<Row> prepareGridDataset(Dataset<Row> df) {
        df = df.withColumn("timestamp", unix_timestamp(col("DataC"), "yyyy-MM-dd"));

        df = df
                .withColumn("OcenaC", col("OcenaC").cast("double"))
                .withColumn("OcenaCpp", col("OcenaCpp").cast("double"));

        return df;
    }

    private static VectorAssembler configVectorAssembler() {
        return new VectorAssembler().setInputCols(FEATURE_NAMES).setOutputCol("features");
    }

    private static LogisticRegression configureLogisticRegression(int iteration, double regParam, double elasticNetParam) {
        return new LogisticRegression()
            .setMaxIter(iteration)
            .setRegParam(regParam)
            .setElasticNetParam(elasticNetParam)
            .setFeaturesCol("features")
            .setLabelCol("Wynik");
    }

    private static LogisticRegressionModel buildLogisticRegressionModel(LogisticRegression lr, Dataset<Row> df) {
        return lr.fit(df);
    }

    private static void printLogisticRegressionEquation(LogisticRegressionModel lrModel) {
        StringBuilder equation = new StringBuilder("logit(zdal) = ");
        double[] coefficients = lrModel.coefficients().toArray();
        double intercept = lrModel.intercept();

        for (int i = 0; i < coefficients.length; i++) {
            double roundedCoefficient = Math.round(coefficients[i] * 1e6) / 1e6;
            equation.append(roundedCoefficient).append("*").append(FEATURE_NAMES[i]).append(" + ");
        }

        System.out.println(equation.toString().concat(String.valueOf(intercept)));
    }

    private static Dataset<Row> calcPredictions(LogisticRegressionModel lrModel, Dataset<Row> df) {
        Dataset<Row> df_with_predictions = lrModel.transform(df);
        return df_with_predictions.select("features","rawPrediction", "probability", "prediction");
    }

    private static void analyzePredictions(Dataset<Row> dfPredictions, LogisticRegressionModel lrModel) {
        double[] coefficients = lrModel.coefficients().toArray();
        double intercept = lrModel.intercept();

        dfPredictions.foreach((ForeachFunction<Row>) row -> {
            Vector features = row.getAs("features");
            Vector rawPrediction = row.getAs("rawPrediction");
            Vector probability = row.getAs("probability");

            double logit = intercept;

            for (int i = 0; i < coefficients.length; i++) {
                logit += coefficients[i] * features.apply(i);
            }

            double p1 = 1 / (1 + Math.exp(-logit));
            double p0 = 1 - p1;

            System.out.println("Logit: " + logit);
            System.out.println("Calculated probability P(0): " + p0);
            System.out.println("Calculated probability P(1): " + p1);
            System.out.println("RawPrediction from model: " + rawPrediction);
            System.out.println("Probability from model: " + probability);

            int predictedClass = probability.argmax();
            double predictedProbability = probability.apply(predictedClass);
            System.out.println("Predicted class: " + predictedClass);
            System.out.println("Predicted probability: " + predictedProbability);
            System.out.println("------------------------------------------------");
        });
    }

    private static Dataset<Row> concatPredictionAndProbColumn(Dataset<Row> df, Dataset<Row> df_predictions) {
        Dataset<Row> dfWithProb = df_predictions
                .withColumn("prob", callUDF("max_vector_element", col("probability")))
                .select(col("prediction"), col("prob"));;

        Dataset<Row> indexedDf = df.withColumn("index", monotonically_increasing_id());
        Dataset<Row> indexedDfWithPredictions = dfWithProb.withColumn("index", monotonically_increasing_id());

        return indexedDf
            .join(indexedDfWithPredictions, indexedDf.col("index").equalTo(indexedDfWithPredictions.col("index")))
            .drop(indexedDfWithPredictions.col("index"))
            .drop(indexedDf.col("index"));
    }

    private static void saveFile(Dataset<Row> df, String filename) {
        df = df.repartition(1);
        df.write()
            .format("csv")
            .option("header", true)
            .option("delimiter", ",")
            .mode(SaveMode.Overwrite)
            .save("results/" + filename);
    }

    private static Object[] trainAndTest(Dataset<Row> df){
        int splitSeed = 123;
        Dataset<Row>[] splits = df.randomSplit(new double[] {0.7, 0.3}, splitSeed);
        Dataset<Row> df_train = splits[0];
        Dataset<Row> df_test = splits[1];

        LogisticRegression lr = new LogisticRegression()
                .setMaxIter(20)
                .setRegParam(0.1)
                .setFeaturesCol("features")
                .setLabelCol("Wynik");

        LogisticRegressionModel model = lr.fit(df_train);
        return new Object[] {model, df_test};
    }

    private static BinaryLogisticRegressionTrainingSummary calcSummary(LogisticRegressionModel lrModel) {
        return lrModel.binarySummary();
    }

    private static double[] calcObjectiveHistory(BinaryLogisticRegressionTrainingSummary trainingSummary) {
        return trainingSummary.objectiveHistory();
    }

    private static void plotObjectiveHistory(double[] objectiveHistory) {
        int[] arrRange = IntStream.rangeClosed(1, objectiveHistory.length).toArray();
        List<Integer> X = Arrays.stream(arrRange).boxed().collect(Collectors.toList());
        List<Double> y = Arrays.stream(objectiveHistory).boxed().collect(Collectors.toList());

        Plot plt = Plot.create();
        plt.plot().add(X).add(y).linestyle("-");
        plt.legend();
        plt.title("Objective history");

        try {
            plt.show();
        } catch (IOException | PythonExecutionException e) {
            throw new RuntimeException(e);
        }
    }

    private static void plotROC(Dataset<Row> roc) {
        List<Double> fpr = roc.select("FPR").as(Encoders.DOUBLE()).collectAsList();
        List<Double> tpr = roc.select("TPR").as(Encoders.DOUBLE()).collectAsList();

        Plot plt = Plot.create();
        plt.plot().add(fpr).add(tpr).linestyle("-");
        plt.legend();
        plt.title("ROC Curve");

        try {
            plt.show();
        } catch (IOException | PythonExecutionException e) {
            throw new RuntimeException(e);
        }
    }

    private static void showOtherMetrics(BinaryLogisticRegressionTrainingSummary trainingSummary) {
        double accuracy = trainingSummary.accuracy();
        double[] truePositiveRateByLabel = trainingSummary.truePositiveRateByLabel();
        double[] falsePositiveRateByLabel = trainingSummary.falsePositiveRateByLabel();
        double[] precisionByLabel = trainingSummary.precisionByLabel();
        double[] recallByLabel = trainingSummary.recallByLabel();
        double[] fMeasureByLabel = trainingSummary.fMeasureByLabel();

        System.out.println("Accuracy: " + accuracy);
        System.out.println("Class 0 Metrics:");
        System.out.println("  TPR (True Positive Rate): " + truePositiveRateByLabel[0]);
        System.out.println("  FPR (False Positive Rate): " + falsePositiveRateByLabel[0]);
        System.out.println("  Precision: " + precisionByLabel[0]);
        System.out.println("  Recall: " + recallByLabel[0]);
        System.out.println("  F-measure: " + fMeasureByLabel[0]);

        System.out.println("Class 1 Metrics:");
        System.out.println("  TPR (True Positive Rate): " + truePositiveRateByLabel[1]);
        System.out.println("  FPR (False Positive Rate): " + falsePositiveRateByLabel[1]);
        System.out.println("  Precision: " + precisionByLabel[1]);
        System.out.println("  Recall: " + recallByLabel[1]);
        System.out.println("  F-measure: " + fMeasureByLabel[1]);
    }

    private static double calcMaxFMeasure(Dataset<Row> dfFMeasures) {
        return dfFMeasures.select(functions.max("F-Measure")).head().getDouble(0);
    }

    private static double findThresholdBaseOnMaxFMeasure(Dataset<Row> dfFMeasures, double FMeasure) {
        return dfFMeasures
            .where(dfFMeasures.col("F-Measure").equalTo(FMeasure))
            .select("threshold")
            .head()
            .getDouble(0);
    }

    private static Dataset<Row> makePredictions(LogisticRegressionModel lrModel, Dataset<Row> df) {
        return lrModel.transform(df);
    }

    private static MulticlassClassificationEvaluator configEvaluator() {
        return new MulticlassClassificationEvaluator()
                .setLabelCol("Wynik")
                .setPredictionCol("prediction");
    }

    private static void calcOtherMetricsBaseOnEvaluatorAndPredictions(
            MulticlassClassificationEvaluator evaluator, Dataset<Row> predictions)
    {
        String[] metrics = {"accuracy", "weightedPrecision", "weightedRecall", "f1"};

        for (String metric : metrics) {
            evaluator.setMetricName(metric);
            double result = evaluator.evaluate(predictions);
            System.out.println(metric + ": " + String.format("%.3f", result));
        }
    }

    public static Dataset<Row> convertPredictionNames(Dataset<Row> df) {
        df = df.drop("timestamp").drop("prob");
        df =  df
                .withColumn("Wynik", when(col("prediction").equalTo(1), "Zdał")
                .otherwise("Nie zdał"))
                .drop("prediction");

        return df;
    }

    public static void addClassificationToGrid(SparkSession spark, LogisticRegressionModel lrModel) {
        spark.udf().register("max_vector_element", new MaxVectorElement(), DataTypes.DoubleType);

        Dataset<Row> df = loadDataframe(spark, DATA_FILE_PATH + "grid.csv", ",");
        Dataset<Row> df_grid = prepareGridDataset(df);

        VectorAssembler va = configVectorAssembler();
        Dataset<Row> df_grid_va = va.transform(df_grid);
        Dataset<Row> df_with_predictions = calcPredictions(lrModel, df_grid_va);
        Dataset<Row> df_with_prob = concatPredictionAndProbColumn(df_grid, df_with_predictions);
        Dataset<Row> df_result = convertPredictionNames(df_with_prob);

        showDataframeContentAndSchema(df_result);
//        saveFile(df_result, "grid-with-classification.csv");
    }

    public static class MaxVectorElement implements UDF1<Vector, Double> {
        @Override
        public Double call(Vector vector) throws Exception {
            return vector.apply(vector.argmax());
        }
    }

}
