package org.example;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.CountVectorizer;
import org.apache.spark.ml.feature.CountVectorizerModel;
import org.apache.spark.ml.feature.RegexTokenizer;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import scala.collection.Iterable;

import java.util.Arrays;

public class AuthorRecognitionCVGridSearch {

    private static final String FILES_PATH = "src/main/resources/";
    private static final String CSV_FORMAT = "csv";
    private static final String SEPARATORS = "[\\s\\p{Punct}—…”„]+";
    private static final String[] FILENAMES_ARR = {
        "two-books-all-1000-1-stem.csv",
        "two-books-all-1000-3-stem.csv",
        "two-books-all-1000-5-stem.csv",
        "two-books-all-1000-10-stem.csv",
        "five-books-all-1000-1-stem.csv",
        "five-books-all-1000-3-stem.csv",
        "five-books-all-1000-5-stem.csv",
        "five-books-all-1000-10-stem.csv",
    };

    public static void main(String[] args) {
        SparkSession spark = createSparkSession();
        performCV(spark, "two-books-all-1000-1-stem.csv");
//        for (String file : FILENAMES_ARR) {
//            performCV(spark, file);
//        }
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

    private static RegexTokenizer getRegexTokenizer() {
        return new RegexTokenizer().setInputCol("content").setOutputCol("words").setPattern(SEPARATORS);
    }

    private static CountVectorizer getCountVectorizer(int vocabSize, int minDF) {
        return new CountVectorizer()
                .setInputCol("words")
                .setOutputCol("features")
                .setVocabSize(vocabSize)
                .setMinDF(minDF);
    }

    private static StringIndexer getStringIndexer() {
        return new StringIndexer().setInputCol("author").setOutputCol("label");
    }

    private static DecisionTreeClassifier getDecisionTreeClassifier() {
        return new DecisionTreeClassifier()
                .setLabelCol("label")
                .setFeaturesCol("features")
                .setImpurity("gini")
                .setMaxDepth(30);
    }

    private static MulticlassClassificationEvaluator getMulticlassEvaluator() {
        return new MulticlassClassificationEvaluator()
            .setLabelCol("label")
            .setPredictionCol("prediction");
    }

    // For Grid Search add parameter paramGrid and use it in param maps
    private static CrossValidator getCrossValidator(
            Pipeline pipeline,
            MulticlassClassificationEvaluator evaluator
    ) {
        return new CrossValidator()
            .setEstimator(pipeline)
            .setEvaluator(evaluator)
            .setEstimatorParamMaps(new ParamGridBuilder().build())
            .setNumFolds(3)
            .setParallelism(8);
    }

    private static Integer getVocabSize(PipelineModel bestModel) {
        CountVectorizerModel countVectorizerModel = (CountVectorizerModel) bestModel.stages()[1];
        return countVectorizerModel.getVocabSize();
    }

    private static Integer getMaxDepth(PipelineModel bestModel) {
        DecisionTreeClassificationModel decisionTreeModel = (DecisionTreeClassificationModel) bestModel.stages()[3];
        return decisionTreeModel.getMaxDepth();
    }

    private static void performCV(SparkSession spark, String filename) {
        Dataset<Row> df = loadFile(spark, filename);
        Dataset<Row>[] splits = df.randomSplit(new double[] {0.8, 0.2});
        Dataset<Row> df_train = splits[0];
        Dataset<Row> df_test = splits[1];

        RegexTokenizer tokenizer = getRegexTokenizer();
        CountVectorizer countVectorizer = getCountVectorizer(10_000, 30);
        StringIndexer labelIndexer = getStringIndexer();
//        DecisionTreeClassifier dt = getDecisionTreeClassifier();
        NaiveBayes nb = getNaiveBayesClassifier();

        PipelineStage[] pipelineStages = new PipelineStage[] {tokenizer, countVectorizer, labelIndexer, nb};
        Pipeline pipeline = new Pipeline().setStages(pipelineStages);

        MulticlassClassificationEvaluator evaluator = getMulticlassEvaluator();
//        ParamMap[] paramGrid = configureParamGrid(countVectorizer);

        CrossValidator cv = getCrossValidator(pipeline, evaluator);
        CrossValidatorModel cvModel = cv.fit(df_train);

        PipelineModel bestModel = (PipelineModel) cvModel.bestModel();

        System.out.println("File name:" + filename);

//        for (var s : bestModel.stages()){
//            System.out.println(s);
//        }

        Dataset<Row> predictions = bestModel.transform(df_test);
        double[] f1AverageMetrics = cvModel.avgMetrics();

        getModelStatistics(predictions, evaluator);

        System.out.println("Średnie wartości metryki F1 dla badanych modeli:");
        for (double metric : f1AverageMetrics) {
            System.out.println(metric);
        }

        System.out.println("-----------------------------");
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

    private static NaiveBayes getNaiveBayesClassifier() {
        return new NaiveBayes()
            .setLabelCol("label")
            .setFeaturesCol("features")
            .setSmoothing(0.2);
    }

    private static Iterable<String> configureScalaIterable() {
        return scala
            .jdk.CollectionConverters
            .IterableHasAsScala(Arrays.asList("multinomial", "gaussian"))
            .asScala();
    }

    private static ParamMap[] configureParamGrid(CountVectorizer countVectorizer) {
        NaiveBayes nb = getNaiveBayesClassifier();
        Iterable<String> scalaIterable = configureScalaIterable();

        return new ParamGridBuilder()
            .addGrid(countVectorizer.vocabSize(), new int[] {100, 1000, 5_000, 10_000})
            .addGrid(nb.modelType(), scalaIterable)
            .build();
    }
}
