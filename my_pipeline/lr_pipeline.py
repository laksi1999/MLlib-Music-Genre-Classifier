from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.functions import lit
from pyspark.ml import Pipeline
from pyspark.ml.feature import StopWordsRemover, Tokenizer, Word2Vec
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, CrossValidatorModel, ParamGridBuilder
from my_pipeline.cleanser import Cleanser
from my_pipeline.encoder import Encoder
from my_pipeline.stemmer import Stemmer


class LRPipeline:
    def __init__(self) -> None:
        self.spark = SparkSession.builder.appName("MLlib-Pipeline")\
            .config("spark.driver.memory", "3G").config("spark.executor.memory", "3G")\
            .config("spark.executor.cores", "3").config("spark.python.worker.memory", "3G")\
            .config("spark.driver.port", "4040").getOrCreate()

    def stop_pipeline(self) -> None:
        self.spark.stop()

    @staticmethod
    def train(dataframe: DataFrame) -> CrossValidatorModel:
        dataframe: DataFrame = dataframe.select("lyrics", "genre")

        pipeline = Pipeline(
            stages=[
                Encoder(),
                Cleanser(),
                Tokenizer(inputCol="clean", outputCol="tokens"),
                StopWordsRemover(inputCol="tokens", outputCol="after_stop_words_removal"),
                Stemmer(),
                Word2Vec(inputCol="stemmed_lyrics", outputCol="features", vectorSize=400, minCount=0, seed=32),
                LogisticRegression(regParam=0.01),
            ]
        )

        cross_validator = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=ParamGridBuilder().build(),
            evaluator=MulticlassClassificationEvaluator(),
            numFolds=5,
            seed=32
        )

        cross_validator_model: CrossValidatorModel = cross_validator.fit(dataframe)

        return cross_validator_model

    @staticmethod
    def test(dataframe: DataFrame, model: CrossValidatorModel) -> float:
        return MulticlassClassificationEvaluator()\
            .evaluate(model.bestModel.transform(dataframe))

    def train_and_test(
        self, dataset_path: str, train_ratio: float, model_storage_path: str
    ) -> CrossValidatorModel:
        data = self.spark.read.csv(dataset_path, header=True, inferSchema=True)
        train_df, test_df = data.randomSplit([train_ratio, (1 - train_ratio)], seed=32)

        model: CrossValidatorModel = self.train(train_df)

        print(f"avgMetrics: {model.avgMetrics}")
        print(f"test_f1: {self.test(test_df, model)}")

        model.write().overwrite().save(model_storage_path)

        return model

    def predict_for_unknown_lyrics(self, unknown_lyrics: str, model: CrossValidatorModel):
        unknown_lyrics_df = self.spark.createDataFrame([(unknown_lyrics,)], ["lyrics"])
        unknown_lyrics_df = unknown_lyrics_df.withColumn("genre", lit("unknown"))

        predictions_df = model.bestModel.transform(unknown_lyrics_df)
        prediction_row = predictions_df.first()

        label_to_genre = [
            "pop", "country", "blues", "rock", "jazz", "reggae", "hip hop", "soul", "unknown",
        ]

        prediction = label_to_genre[int(prediction_row["prediction"])]
        probabilities = dict(zip(label_to_genre, prediction_row["probability"]))

        return prediction, probabilities
