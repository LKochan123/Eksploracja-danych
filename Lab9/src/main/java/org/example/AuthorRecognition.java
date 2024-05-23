package org.example;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.linalg.SparseVector;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.*;

import java.util.Arrays;

public class AuthorRecognition {

    private static final String FILES_PATH = "src/main/resources/";
    private static final String CSV_FORMAT = "csv";
    private static final String SEPARATORS = "[\\s\\p{Punct}—…”„]+";
    public static void main(String[] args) {
        SparkSession spark = createSparkSession();
        Dataset<Row> df = loadFile(spark, "two-books-all-1000-10-stem.csv");
        Dataset<Row> df_author_work = selectAppropriateColumns(df, "author", "work");
        Dataset<Row> dfTokenized = tokenizeDataframe(df);
        Dataset<Row> dfBow = getDataframeAsBagOfWords(dfTokenized);
        Dataset<Row> dfBowTransformed = transformAuthorNamesToNumbers(dfBow);

        DecisionTreeClassificationModel model = getDTCModel(dfBowTransformed);
        Dataset<Row> dfPredictions = makePredictions(dfBowTransformed);

        SparseVector fi = getFeatureImportance(model);

        CountVectorizerModel countVectorizerModel = getCountVectorizerModel(dfTokenized);
//        showWordsOccurrence(dfBow, countVectorizerModel);
//        getModelAccuracy(dfPredictions);
//        showFeaturesImportance(model);

//        calcAuthorDocuments(df);
//        averageContentLength(df);
//        tokenizeDataframe(df);
//        showDataframeContent(dfPredictions);

//        showWordImportance(fi, countVectorizerModel);
//        showWordsAndFeature(dfBow);

        // 3. AuthorRecognitionGridSearchCVDecisionTree
        performGridSearchCV(spark, "two-books-all-1000-10-stem.csv");
    }

    private static SparkSession createSparkSession() {
        return SparkSession.builder()
            .appName("LogisticRegressionOnExam")
            .master("local")
            .getOrCreate();
    }

    private static Dataset<Row> loadFile(SparkSession spark, String fileName) {
        return spark.read().format(CSV_FORMAT)
            .option("header", "true")
            .option("delimiter", ",")
            .option("quote", "\'")
            .option("inferschema", "true")
            .load(FILES_PATH + fileName);
    }

    private static void showDataframeContent(Dataset<Row> df) {
        System.out.println("Excerpt of the dataframe content:");
        df.show(10);

        System.out.println("Dataframe's schema:");
        df.printSchema();
    }

    private static Dataset<Row> selectAppropriateColumns(Dataset<Row> df, String ...columns) {
        Column[] cols = Arrays.stream(columns).map(functions::col).toArray(Column[]::new);
        return df.select(cols).distinct();
    }

    private static void calcAuthorDocuments(Dataset<Row> df) {
       showDataframeContent(df.groupBy("author").count());
    }

    private static void averageContentLength(Dataset<Row> df) {
        Dataset<Row> df_length = df.withColumn("contentLength", functions.length(df.col("content")));
        Dataset<Row> df_avg = df_length
                .groupBy("author")
                .avg("contentLength")
                .withColumnRenamed("avg(contentLength)", "averageContentLength");

        showDataframeContent(df_avg);
    }

    private static RegexTokenizer getRegexTokenizer() {
        return new RegexTokenizer().setInputCol("content").setOutputCol("words").setPattern(SEPARATORS);
    }

    private static Dataset<Row> tokenizeDataframe(Dataset<Row> df) {
        RegexTokenizer tokenizer = getRegexTokenizer();
        return tokenizer.transform(df);
    }

    private static CountVectorizer getCountVectorizer(int vocabSize, int minDF) {
        return new CountVectorizer()
            .setInputCol("words")
            .setOutputCol("features")
            .setVocabSize(vocabSize)
            .setMinDF(minDF);
    }

    private static CountVectorizerModel getCountVectorizerModel(Dataset<Row> dfTokenized) {
        CountVectorizer countVectorizer = getCountVectorizer(10_000, 2);
        return countVectorizer.fit(dfTokenized);
    }

    private static Dataset<Row> getDataframeAsBagOfWords(Dataset<Row> dfTokenized) {
        CountVectorizerModel countVectorizerModel = getCountVectorizerModel(dfTokenized);
        return countVectorizerModel.transform(dfTokenized);
    }

    private static void showWordsAndFeature(Dataset<Row> dfBow) {
        Row row = dfBow.select("words", "features").first();
        System.out.println(row.get(0));
        System.out.println(row.get(1));
    }

    private static void showWordsOccurrence(Dataset<Row> dfBow, CountVectorizerModel countVectorizerModel) {
        Row row = dfBow.select("words", "features").first();
        SparseVector vector = (SparseVector) row.get(1);
        int[] indices = vector.indices();
        double[] values = vector.values();
        String[] words = countVectorizerModel.vocabulary();

        for (int idx : indices) {
            System.out.println(words[idx] + " -> " + values[idx]);
        }
    }

    private static StringIndexer getStringIndexer() {
        return new StringIndexer().setInputCol("author").setOutputCol("label");
    }

    private static Dataset<Row> transformAuthorNamesToNumbers(Dataset<Row> dfBow) {
        StringIndexer labelIndexer = getStringIndexer();
        StringIndexerModel labelModel = labelIndexer.fit(dfBow);
        Dataset<Row> dfBowTransformed = labelModel.transform(dfBow);
        return dfBowTransformed.drop("author", "work", "content", "content_stemmed");
    }

    private static DecisionTreeClassifier getDecisionTreeClassifier() {
        return new DecisionTreeClassifier()
            .setLabelCol("label")
            .setFeaturesCol("features")
            .setImpurity("gini")
            .setMaxDepth(30);
    }

    private static DecisionTreeClassificationModel getDTCModel(Dataset<Row> dfBow) {
        DecisionTreeClassifier dt = getDecisionTreeClassifier();
        return dt.fit(dfBow);
    }

    private static Dataset<Row> makePredictions(Dataset<Row> dfBow) {
        DecisionTreeClassificationModel dtModel = getDTCModel(dfBow);
        return dtModel.transform(dfBow);
    }

    private static MulticlassClassificationEvaluator getEvaluator() {
        return new MulticlassClassificationEvaluator()
            .setLabelCol("label")
            .setPredictionCol("prediction");
    }

    private static void getModelAccuracy(Dataset<Row> predictions) {
        MulticlassClassificationEvaluator evaluator = getEvaluator();

        evaluator.setMetricName("accuracy");
        double accuracy = evaluator.evaluate(predictions);

        evaluator.setMetricName("f1");
        double f1 = evaluator.evaluate(predictions);

        System.out.println("Accuracy: " + accuracy);
        System.out.println("F1 score: " + f1);
    }

    public static SparseVector getFeatureImportance(DecisionTreeClassificationModel model) {
        return (SparseVector) model.featureImportances();
    }

    // Something is wrong here
    public static void showWordImportance(SparseVector fi, CountVectorizerModel model) {
        String[] words = model.vocabulary();
        int[] indices = fi.indices();
        double[] values = fi.values();

        for (int idx : indices) {
            System.out.println(words[idx] + " -> " + values[idx]);
        }
    }

    private static void performGridSearchCV(SparkSession spark, String filename) {
        Dataset<Row> df = loadFile(spark, filename);
        Dataset<Row>[] splits = df.randomSplit(new double[] {0.8, 0.2});
        Dataset<Row> df_train = splits[0];
        Dataset<Row> df_test = splits[1];

        RegexTokenizer tokenizer = getRegexTokenizer();
        CountVectorizer countVectorizer = getCountVectorizer(10_000, 2);
        StringIndexer labelIndexer = getStringIndexer();
        DecisionTreeClassifier dt = getDecisionTreeClassifier();

        PipelineStage[] pipelineStages = new PipelineStage[] {tokenizer, countVectorizer, labelIndexer, dt};
        Pipeline pipeline = new Pipeline().setStages(pipelineStages);

        ParamMap[] paramGrid = new ParamGridBuilder()
            .addGrid(countVectorizer.vocabSize(), new int[] {100, 1000, 10_000})
            .addGrid(dt.maxDepth(), new int[] {10, 20, 30})
            .build();

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
            .setLabelCol("label")
            .setPredictionCol("prediction")
            .setMetricName("f1");

        CrossValidator cv = new CrossValidator()
            .setEstimator(pipeline)
            .setEvaluator(evaluator)
            .setEstimatorParamMaps(paramGrid)
            .setNumFolds(3)
            .setParallelism(8);

        CrossValidatorModel cvModel = cv.fit(df_train);
        PipelineModel bestModel = (PipelineModel) cvModel.bestModel();

        for (var s : bestModel.stages()){
            System.out.println(s);
        }

        Dataset<Row> predictions = bestModel.transform(df_test);
        double[] f1AverageMetrics = cvModel.avgMetrics();

        getModelStatistics(predictions, evaluator);
        System.out.println("-----------------------------");

        System.out.println("Średnie wartości metryki F1 dla badanych modeli:");
        for (double metric : f1AverageMetrics) {
            System.out.println(metric);
        }

    }

    private static void getModelStatistics(Dataset<Row> predictions, MulticlassClassificationEvaluator evaluator) {
        evaluator.setMetricName("accuracy");
        double accuracy = evaluator.evaluate(predictions);

        evaluator.setMetricName("f1");
        double f1 = evaluator.evaluate(predictions);

        evaluator.setMetricName("weightedPrecision");
        double weightedPrecision = evaluator.evaluate(predictions);

        evaluator.setMetricName("weightedRecall");
        double weightedRecall = evaluator.evaluate(predictions);

        System.out.println("Accuracy: " + accuracy);
        System.out.println("F1 score: " + f1);
        System.out.println("Weighted precision" + weightedPrecision);
        System.out.println("Weighted recall" + weightedRecall);
    }

}
