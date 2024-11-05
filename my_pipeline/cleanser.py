from pyspark.sql import DataFrame
from pyspark.sql.functions import col, regexp_replace, trim
from my_pipeline.my_transformer import MyTransformer


class Cleanser(MyTransformer):
    def _transform(self, dataframe: DataFrame) -> DataFrame:
        dataframe = dataframe.withColumn("clean", regexp_replace(trim(col("lyrics")), r"[^\w\s]", ""))\
            .withColumn("clean", regexp_replace(col("clean"), r"\s{2,}", " "))
        return dataframe.filter(col("clean").isNotNull())
