package org.example;

import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.linalg.Matrix;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;

import java.util.Arrays;
import java.util.List;
import java.util.Locale;

public class NaiveBayesDemo {

    private static final String SEPARATORS = "[\\s\\p{Punct}—…”„]+";
    public static void main(String[] args) {
        SparkSession spark = createSparkSession();
        Dataset<Row> df = getDataset(spark);

        NaiveBayesModel model = processCV(df);
        makePredictions(model);
    }

    private static SparkSession createSparkSession() {
        return SparkSession.builder()
            .appName("LogisticRegressionOnExam")
            .master("local")
            .getOrCreate();
    }

    private static void showDataframeContent(Dataset<Row> df) {
        System.out.println("Excerpt of the dataframe content:");
        df.show(10);

        System.out.println("Dataframe's schema:");
        df.printSchema();
    }

    private static StructType getSchema() {
        return new StructType()
            .add("author", DataTypes.StringType, false)
            .add("content", DataTypes.StringType, false);
    }

    private static List<Row> getSomeTestData() {
        return Arrays.asList(
            RowFactory.create("Ala","aaa aaa bbb ccc"),
            RowFactory.create("Ala","aaa bbb ddd"),
            RowFactory.create("Ala","aaa bbb"),
            RowFactory.create("Ala","aaa bbb bbb"),
            RowFactory.create("Ola","aaa ccc ddd"),
            RowFactory.create("Ola","bbb ccc ddd"),
            RowFactory.create("Ola","ccc ddd eee")
        );
    }

    private static Dataset<Row> getDataset(SparkSession spark) {
        StructType schema = getSchema();
        List<Row> rows = getSomeTestData();
        return spark.createDataFrame(rows, schema);
    }

    private static RegexTokenizer getRegexTokenizer() {
        return new RegexTokenizer()
            .setInputCol("content")
            .setOutputCol("words")
            .setPattern(SEPARATORS);
    }

    private static Dataset<Row> dataframeTransform(RegexTokenizer tokenizer, Dataset<Row> df) {
        return tokenizer.transform(df);
    }

    private static Dataset<Row> dataframeTransform(CountVectorizerModel countVectorizerModel, Dataset<Row> df) {
        return countVectorizerModel.transform(df);
    }

    private static Dataset<Row> dataframeTransform(StringIndexerModel labelModel, Dataset<Row> df) {
        return labelModel.transform(df);
    }

    private static CountVectorizer getCountVectorizer() {
        return new CountVectorizer()
            .setInputCol("words")
            .setOutputCol("features")
            .setVocabSize(10_000)
            .setMinDF(1);
    }

    private static CountVectorizerModel getCountVectorizerModel(CountVectorizer countVectorizer, Dataset<Row> df) {
        return countVectorizer.fit(df);
    }

    private static StringIndexer getStringIndexer() {
        return new StringIndexer()
            .setInputCol("author")
            .setOutputCol("label");
    }

    private static StringIndexerModel getStringIndexerModel(StringIndexer labelIndexer, Dataset<Row> df) {
        return labelIndexer.fit(df);
    }

    private static NaiveBayes getNaiveBayes() {
        return new NaiveBayes()
            .setLabelCol("label")
            .setFeaturesCol("features")
            .setModelType("multinomial")
            .setSmoothing(0.01);
    }

    private static NaiveBayesModel getNaiveBayesModel(Dataset<Row> df) {
        NaiveBayes nb = getNaiveBayes();
        return nb.fit(df);
    }

    private static NaiveBayesModel processCV(Dataset<Row> df) {
        RegexTokenizer tokenizer = getRegexTokenizer();
        Dataset<Row> dfTokenizer = dataframeTransform(tokenizer, df);

        CountVectorizer countVectorizer = getCountVectorizer();
        CountVectorizerModel countVectorizerModel = getCountVectorizerModel(countVectorizer, dfTokenizer);
        Dataset<Row> dfCV = dataframeTransform(countVectorizerModel, dfTokenizer);

        StringIndexer labelIndexer = getStringIndexer();
        StringIndexerModel labelModel = getStringIndexerModel(labelIndexer, dfCV);
        Dataset<Row> dfProcessed = dataframeTransform(labelModel, dfCV);

        NaiveBayesModel model = getNaiveBayesModel(dfProcessed);
        showConditionalProbability(model, countVectorizerModel, labelModel);

        return model;
    }

    private static void showConditionalProbability(
            NaiveBayesModel model,
            CountVectorizerModel countVectorizerModel,
            StringIndexerModel labelModel
    ) {
        String[] vocabs = countVectorizerModel.vocabulary();
        String[] labels = labelModel.labels();

        Matrix theta = model.theta();
        Vector likelihood = model.pi();

        System.out.println(theta);
        System.out.println(likelihood);

        for (int i = 0; i < theta.numRows(); i++) {
            for (int j = 0; j < theta.numCols(); j++) {
                double log = theta.apply(i, j);
                String label = labels[i];
                String vocab = vocabs[j];

                System.out.println("P(" + vocab + "|" + label + ")=" + Math.exp(log) + " (log=" + log + ")");
            }
        }

        showAPrioriProbability(likelihood, labels);
    }

    private static void showAPrioriProbability(Vector likelihood, String[] labels) {
        for (int i = 0; i < likelihood.size(); i++) {
            String label = labels[i];
            double probability = likelihood.apply(i);
            System.out.println("P(" + label + ")=" + Math.exp(probability) + " (log=" + probability + ")");
        }
    }

    private static void makePredictions(NaiveBayesModel model) {
        DenseVector testData = new DenseVector(new double[] {1, 0, 2, 1, 0});
        Vector predictions = model.predictRaw(testData);

        double p0 = Math.exp(predictions.apply(0));
        double p1 = Math.exp(predictions.apply(1));

        System.out.println("Pr:[" + p0 + ", " + p1);
        double predictionsLabel = model.predict(testData);
        System.out.println(predictionsLabel);

        calcP0P1Likelihoods(p0, p1);
    }

    private static void calcP0P1Likelihoods(double p0, double p1) {
        System.out.printf(Locale.US, "log(p0)=%g p0=%g log(p1)=%g p1=%g\n", p0, Math.exp(p0), p1, Math.exp(p1));
        System.out.println("Wynik klasyfikacj:" + (p0 > p1 ? 0 : 1));
    }
}
