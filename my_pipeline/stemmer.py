from nltk.stem import SnowballStemmer
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, StringType
from my_pipeline.my_transformer import MyTransformer


class Stemmer(MyTransformer):
    def _transform(self, dataframe: DataFrame) -> DataFrame:
        stemmer = SnowballStemmer("english")
        stem_udf = udf(lambda words: [stemmer.stem(w) for w in words], ArrayType(StringType()))
        return dataframe.withColumn("stemmed_lyrics", stem_udf(col("after_stop_words_removal")))
