from datetime import datetime
import logging
import functools as fn
import pandas as pd

from prometheus_api_client import PrometheusConnect, MetricRangeDataFrame

# Initialize logger
logger = logging.getLogger(__name__)


class Query:
    def __init__(self, query, cols=[]):
        self.query = query
        self.cols = cols

    def retain(self, cols=None):
        self.cols = cols
        return self

    def rate(self, step="20s"):
        return Query(query=f"rate( {self.query}[{step}] )", cols=self.cols)

    def sum(self, by=None):
        if by is not None:
            return Query(query=f"sum by({''.join(by)})({self.query})", cols=self.cols + by)

        return Query(query=f"sum( {self.query} )", cols=self.cols)


class Client:
    def __init__(self, url: str):
        self.prom = PrometheusConnect(url, disable_ssl=True)

    def instant_query(self, at: datetime, **queries):
        results = []
        common_columns = set(["timestamp"])

        # Iterate over each query provided in kwargs
        for key, promql in queries.items():
            logger.debug(f"Running query {key}: '{promql}' @  {at}")

            data = self.prom.custom_query_range(query=promql, start_time=at, end_time=at, step="1s")
            if not data or len(data[0]["values"]) == 0:
                raise ValueError(f"No data found for query: {promql}")

            if len(data) != 1:
                raise ValueError(f"Expected single time-series but got {len(data)} for query: {promql}")

            # common_columns = common_columns.union(set(promql.cols))
            metric_df = MetricRangeDataFrame(
                data=data,
                ts_as_datetime=False,
            )
            metric_df.index = metric_df.index.astype("int64")
            metric_df.rename(columns={"value": key}, inplace=True)
            results.append(metric_df)

        # Merge all DataFrames on timestamp column
        logger.debug("results: %d", len(results))
        merged_df = fn.reduce(
            lambda left, right: pd.merge(left, right, on=list(common_columns), how="inner"),
            results,
        )
        # Optionally, sort the final DataFrame by timestamp if needed
        merged_df.sort_index(inplace=True)
        return merged_df

    def range_query(self, start: datetime, end: datetime, step, **queries):
        results = []
        common_columns = set(["timestamp"])

        # Iterate over each query provided in kwargs
        for key, promql in queries.items():
            logger.info(f"Running query {key}: '{promql}' with step {step} between {start} and {end}")

            data = self.prom.custom_query_range(query=promql, start_time=start, end_time=end, step=step)
            if not data or len(data[0]["values"]) == 0:
                raise ValueError(f"No data found for query: {promql}")

            if len(data) != 1:
                raise ValueError(f"Expected single time-series but got {len(data)} for query: {promql}")

            # common_columns = common_columns.union(set(promql.cols))
            metric_df = MetricRangeDataFrame(
                data=data,
                ts_as_datetime=False,
            )
            metric_df.index = metric_df.index.astype("int64")
            metric_df.rename(columns={"value": key}, inplace=True)
            results.append(metric_df)

        # Merge all DataFrames on timestamp column
        logger.info("results: %d", len(results))
        merged_df = fn.reduce(
            lambda left, right: pd.merge(left, right, on=list(common_columns), how="inner"),
            results,
        )
        # Optionally, sort the final DataFrame by timestamp if needed
        merged_df.sort_index(inplace=True)
        return merged_df
