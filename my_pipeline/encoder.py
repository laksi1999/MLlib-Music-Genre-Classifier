from pyspark.sql import DataFrame
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType
from my_pipeline.my_transformer import MyTransformer


class Encoder(MyTransformer):
    def _transform(self, dataframe: DataFrame) -> DataFrame:
        genre_to_label = {
            "pop": 0,
            "country": 1,
            "blues": 2,
            "rock": 3,
            "jazz": 4,
            "reggae": 5,
            "hip hop": 6,
            "soul": 7,
            "unknown": 8,
        }
        genre_to_label_udf = udf(lambda g: genre_to_label.get(g, 8), IntegerType())
        return dataframe.withColumn("label", genre_to_label_udf(col("genre")))
