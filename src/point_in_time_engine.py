"""为Alpha101提供向量化的逐信号日点时复权计算引擎。"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from types import FunctionType
from typing import Iterator, Mapping

import numpy as np
import pandas as pd

import alphas
from factor_data import FactorData


PRICE_FIELDS = frozenset({"open", "high", "low", "close", "vwap"})
_UNIVERSE_MASK: ContextVar[pd.DataFrame | None] = ContextVar(
    "pit_universe_mask",
    default=None,
)


def _same_axes(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    return left.index.equals(right.index) and left.columns.equals(right.columns)


class PITExpression:
    """按复权次数保存表达式，使滚动运算不必逐日重建历史价格。

    每一项 ``degree -> base`` 表示在日期t的实际值为
    ``base[t] / adj_factor[t] ** degree``。价格字段的degree为1，
    成交量、收益率和排名等无量纲字段的degree为0。
    """

    __array_priority__ = 1000

    def __init__(
        self,
        terms: Mapping[float, pd.DataFrame],
        adj_factor: pd.DataFrame,
    ) -> None:
        if not terms:
            raise ValueError("PITExpression至少需要一项")
        self.adj_factor = adj_factor
        cleaned: dict[float, pd.DataFrame] = {}
        for degree, frame in terms.items():
            if not isinstance(frame, pd.DataFrame):
                raise TypeError("PITExpression项必须是DataFrame")
            if not _same_axes(frame, adj_factor):
                raise ValueError("PITExpression项与复权因子不对齐")
            numeric_degree = float(degree)
            cleaned[numeric_degree] = (
                cleaned[numeric_degree] + frame
                if numeric_degree in cleaned
                else frame
            )
        self.terms = cleaned

    @classmethod
    def from_value(
        cls,
        value: pd.DataFrame,
        adj_factor: pd.DataFrame,
        *,
        degree: float = 0.0,
    ) -> "PITExpression":
        return cls({float(degree): value}, adj_factor)

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.adj_factor.index

    @property
    def columns(self) -> pd.Index:
        return self.adj_factor.columns

    @property
    def shape(self) -> tuple[int, int]:
        return self.adj_factor.shape

    @property
    def value(self) -> pd.DataFrame:
        result: pd.DataFrame | None = None
        for degree, base in self.terms.items():
            term = base if degree == 0 else base / self.adj_factor.pow(degree)
            result = term if result is None else result + term
        if result is None:  # pragma: no cover - constructor prevents this
            raise RuntimeError("PITExpression没有可求值项")
        return result.replace([np.inf, -np.inf], np.nan)

    @property
    def single_term(self) -> tuple[float, pd.DataFrame] | None:
        return next(iter(self.terms.items())) if len(self.terms) == 1 else None

    def _constant(self, value: float) -> "PITExpression":
        frame = pd.DataFrame(value, index=self.index, columns=self.columns)
        return PITExpression.from_value(frame, self.adj_factor)

    def _coerce(self, other: object) -> "PITExpression":
        if isinstance(other, PITExpression):
            if not _same_axes(other.adj_factor, self.adj_factor):
                raise ValueError("PITExpression输入不对齐")
            return other
        if isinstance(other, pd.DataFrame):
            if not _same_axes(other, self.adj_factor):
                raise ValueError("DataFrame输入与PITExpression不对齐")
            return PITExpression.from_value(other, self.adj_factor)
        if np.isscalar(other):
            return self._constant(float(other))
        raise TypeError(f"不支持与PITExpression运算的类型：{type(other).__name__}")

    def _materialized(self, value: pd.DataFrame) -> "PITExpression":
        return PITExpression.from_value(value, self.adj_factor)

    def __add__(self, other: object) -> "PITExpression":
        right = self._coerce(other)
        terms = {degree: frame.copy(deep=False) for degree, frame in self.terms.items()}
        for degree, frame in right.terms.items():
            terms[degree] = terms[degree] + frame if degree in terms else frame
        return PITExpression(terms, self.adj_factor)

    def __radd__(self, other: object) -> "PITExpression":
        return self + other

    def __sub__(self, other: object) -> "PITExpression":
        return self + (-self._coerce(other))

    def __rsub__(self, other: object) -> "PITExpression":
        return self._coerce(other) - self

    def __neg__(self) -> "PITExpression":
        return PITExpression(
            {degree: -frame for degree, frame in self.terms.items()},
            self.adj_factor,
        )

    def __mul__(self, other: object) -> "PITExpression":
        right = self._coerce(other)
        terms: dict[float, pd.DataFrame] = {}
        for left_degree, left_frame in self.terms.items():
            for right_degree, right_frame in right.terms.items():
                degree = left_degree + right_degree
                product = left_frame * right_frame
                terms[degree] = terms[degree] + product if degree in terms else product
        return PITExpression(terms, self.adj_factor)

    def __rmul__(self, other: object) -> "PITExpression":
        return self * other

    def __truediv__(self, other: object) -> "PITExpression":
        right = self._coerce(other)
        denominator = right.single_term
        if denominator is None:
            return self._materialized(self.value / right.value.mask(right.value == 0))
        right_degree, right_base = denominator
        right_base = right_base.mask(right_base == 0)
        return PITExpression(
            {
                degree - right_degree: frame / right_base
                for degree, frame in self.terms.items()
            },
            self.adj_factor,
        )

    def __rtruediv__(self, other: object) -> "PITExpression":
        return self._coerce(other) / self

    def pow(self, exponent: object) -> "PITExpression":
        if isinstance(exponent, PITExpression):
            return self._materialized(self.value.pow(exponent.value))
        if isinstance(exponent, pd.DataFrame):
            return self._materialized(self.value.pow(exponent))
        power = float(exponent)
        single = self.single_term
        if single is not None:
            degree, base = single
            return PITExpression(
                {degree * power: base.pow(power)},
                self.adj_factor,
            )
        if power.is_integer() and 0 <= power <= 10:
            result = self._constant(1.0)
            for _ in range(int(power)):
                result = result * self
            return result
        return self._materialized(self.value.pow(power))

    def __pow__(self, exponent: object) -> "PITExpression":
        return self.pow(exponent)

    def abs(self) -> "PITExpression":
        single = self.single_term
        if single is None:
            return self._materialized(self.value.abs())
        degree, base = single
        return PITExpression({degree: base.abs()}, self.adj_factor)

    def mask(
        self,
        condition: pd.DataFrame,
        other: object = np.nan,
    ) -> "PITExpression":
        if isinstance(other, float) and np.isnan(other):
            return PITExpression(
                {degree: frame.mask(condition) for degree, frame in self.terms.items()},
                self.adj_factor,
            )
        return self._materialized(self.value.mask(condition, other))

    def where(
        self,
        condition: pd.DataFrame,
        other: object = np.nan,
    ) -> "PITExpression":
        if isinstance(other, PITExpression):
            return self._materialized(self.value.where(condition, other.value))
        return self._materialized(self.value.where(condition, other))

    def replace(self, to_replace: object, value: object) -> "PITExpression":
        return PITExpression(
            {
                degree: frame.replace(to_replace, value)
                for degree, frame in self.terms.items()
            },
            self.adj_factor,
        )

    def _compare(self, other: object, operator: str) -> pd.DataFrame:
        right = self._coerce(other).value
        operations = {
            "lt": self.value < right,
            "le": self.value <= right,
            "gt": self.value > right,
            "ge": self.value >= right,
            "eq": self.value == right,
        }
        return operations[operator]

    def __lt__(self, other: object) -> pd.DataFrame:
        return self._compare(other, "lt")

    def __le__(self, other: object) -> pd.DataFrame:
        return self._compare(other, "le")

    def __gt__(self, other: object) -> pd.DataFrame:
        return self._compare(other, "gt")

    def __ge__(self, other: object) -> pd.DataFrame:
        return self._compare(other, "ge")

    def __eq__(self, other: object) -> pd.DataFrame:  # type: ignore[override]
        return self._compare(other, "eq")


def _as_expression(
    value: PITExpression | pd.DataFrame,
    reference: PITExpression | None = None,
) -> PITExpression:
    if isinstance(value, PITExpression):
        return value
    if reference is None:
        raise TypeError("DataFrame转换为PITExpression时需要参考表达式")
    return PITExpression.from_value(value, reference.adj_factor)


class PITOperators:
    """与operators.py同名的点时复权算子集合。"""

    @staticmethod
    def _mask(frame: pd.DataFrame) -> pd.DataFrame:
        mask = _UNIVERSE_MASK.get()
        return frame if mask is None else frame.where(mask)

    def rank(self, data: PITExpression) -> PITExpression:
        value = self._mask(data.value)
        ranked = value.rank(axis=1, method="average", pct=True, na_option="keep")
        return PITExpression.from_value(ranked, data.adj_factor)

    def delay(self, data: PITExpression, period: int = 1) -> PITExpression:
        return PITExpression(
            {degree: frame.shift(period) for degree, frame in data.terms.items()},
            data.adj_factor,
        )

    def delta(self, data: PITExpression, period: int = 1) -> PITExpression:
        return PITExpression(
            {degree: frame.diff(period) for degree, frame in data.terms.items()},
            data.adj_factor,
        )

    def rolling_mean(self, data: PITExpression, window: int) -> PITExpression:
        return PITExpression(
            {
                degree: frame.rolling(window, min_periods=window).mean()
                for degree, frame in data.terms.items()
            },
            data.adj_factor,
        )

    def rolling_sum(self, data: PITExpression, window: int) -> PITExpression:
        return PITExpression(
            {
                degree: frame.rolling(window, min_periods=window).sum()
                for degree, frame in data.terms.items()
            },
            data.adj_factor,
        )

    def rolling_covariance(
        self,
        left: PITExpression,
        right: PITExpression,
        window: int,
    ) -> PITExpression:
        terms: dict[float, pd.DataFrame] = {}
        for left_degree, left_frame in left.terms.items():
            for right_degree, right_frame in right.terms.items():
                degree = left_degree + right_degree
                covariance = left_frame.rolling(
                    window,
                    min_periods=window,
                ).cov(right_frame)
                terms[degree] = (
                    terms[degree] + covariance if degree in terms else covariance
                )
        return PITExpression(terms, left.adj_factor)

    def rolling_std(self, data: PITExpression, window: int) -> PITExpression:
        return self.rolling_covariance(data, data, window).pow(0.5)

    def rolling_correlation(
        self,
        left: PITExpression,
        right: PITExpression,
        window: int,
    ) -> PITExpression:
        covariance = self.rolling_covariance(left, right, window)
        denominator = self.rolling_std(left, window) * self.rolling_std(
            right,
            window,
        )
        return covariance / denominator

    def ts_min(self, data: PITExpression, window: int) -> PITExpression:
        single = data.single_term
        if single is None:
            value = data.value.rolling(window, min_periods=window).min()
            return PITExpression.from_value(value, data.adj_factor)
        degree, base = single
        return PITExpression(
            {degree: base.rolling(window, min_periods=window).min()},
            data.adj_factor,
        )

    def ts_max(self, data: PITExpression, window: int) -> PITExpression:
        single = data.single_term
        if single is None:
            value = data.value.rolling(window, min_periods=window).max()
            return PITExpression.from_value(value, data.adj_factor)
        degree, base = single
        return PITExpression(
            {degree: base.rolling(window, min_periods=window).max()},
            data.adj_factor,
        )

    @staticmethod
    def _rank_last(values: np.ndarray) -> float:
        last = values[-1]
        return float(np.sum(values < last) + (np.sum(values == last) + 1) / 2) / len(values)

    def ts_rank(self, data: PITExpression, window: int) -> PITExpression:
        base = data.single_term[1] if data.single_term is not None else data.value
        value = base.rolling(window, min_periods=window).apply(
            self._rank_last,
            raw=True,
        )
        return PITExpression.from_value(value, data.adj_factor)

    def ts_argmax(self, data: PITExpression, window: int) -> PITExpression:
        base = data.single_term[1] if data.single_term is not None else data.value
        value = base.rolling(window, min_periods=window).apply(
            lambda values: float(np.argmax(values) + 1),
            raw=True,
        )
        return PITExpression.from_value(value, data.adj_factor)

    def ts_argmin(self, data: PITExpression, window: int) -> PITExpression:
        base = data.single_term[1] if data.single_term is not None else data.value
        value = base.rolling(window, min_periods=window).apply(
            lambda values: float(np.argmin(values) + 1),
            raw=True,
        )
        return PITExpression.from_value(value, data.adj_factor)

    def scale(self, data: PITExpression, k: float = 1.0) -> PITExpression:
        value = self._mask(data.value)
        denominator = value.abs().sum(axis=1, min_count=1).replace(0, np.nan)
        return PITExpression.from_value(
            value.mul(k).div(denominator, axis=0),
            data.adj_factor,
        )

    def sign(self, data: PITExpression) -> PITExpression:
        single = data.single_term
        value = np.sign(single[1] if single is not None else data.value)
        return PITExpression.from_value(value, data.adj_factor)

    def abs(self, data: PITExpression) -> PITExpression:
        return data.abs()

    def decay_linear(self, data: PITExpression, window: int) -> PITExpression:
        weights = np.arange(1.0, window + 1.0)
        denominator = weights.sum()

        def apply(frame: pd.DataFrame) -> pd.DataFrame:
            return frame.rolling(window, min_periods=window).apply(
                lambda values: float(np.dot(values, weights) / denominator),
                raw=True,
            )

        return PITExpression(
            {degree: apply(frame) for degree, frame in data.terms.items()},
            data.adj_factor,
        )

    def elementwise_max(
        self,
        left: PITExpression,
        right: PITExpression | float,
    ) -> PITExpression:
        right_expr = left._coerce(right)
        if (
            left.single_term is not None
            and right_expr.single_term is not None
            and left.single_term[0] == right_expr.single_term[0]
        ):
            degree = left.single_term[0]
            value = np.maximum(left.single_term[1], right_expr.single_term[1])
            return PITExpression({degree: value}, left.adj_factor)
        return PITExpression.from_value(
            np.maximum(left.value, right_expr.value),
            left.adj_factor,
        )

    def elementwise_min(
        self,
        left: PITExpression,
        right: PITExpression | float,
    ) -> PITExpression:
        right_expr = left._coerce(right)
        if (
            left.single_term is not None
            and right_expr.single_term is not None
            and left.single_term[0] == right_expr.single_term[0]
        ):
            degree = left.single_term[0]
            value = np.minimum(left.single_term[1], right_expr.single_term[1])
            return PITExpression({degree: value}, left.adj_factor)
        return PITExpression.from_value(
            np.minimum(left.value, right_expr.value),
            left.adj_factor,
        )

    def industry_neutralize(
        self,
        data: PITExpression,
        groups: pd.DataFrame,
    ) -> PITExpression:
        value = self._mask(data.value)
        result = pd.DataFrame(np.nan, index=value.index, columns=value.columns)
        for date in value.index:
            row = value.loc[date]
            group_row = groups.loc[date]
            valid = row.notna() & group_row.notna()
            if valid.any():
                means = row[valid].groupby(group_row[valid]).transform("mean")
                result.loc[date, valid] = row[valid] - means
        return PITExpression.from_value(result, data.adj_factor)

    def log(self, data: PITExpression) -> PITExpression:
        value = np.log(data.value.where(data.value > 0))
        return PITExpression.from_value(value, data.adj_factor)

    def rolling_product(self, data: PITExpression, window: int) -> PITExpression:
        single = data.single_term
        if single is None:
            value = data.value.rolling(window, min_periods=window).apply(
                np.prod,
                raw=True,
            )
            return PITExpression.from_value(value, data.adj_factor)
        degree, base = single
        value = base.rolling(window, min_periods=window).apply(np.prod, raw=True)
        return PITExpression({degree * window: value}, data.adj_factor)

    def signed_power(
        self,
        data: PITExpression,
        exponent: float | pd.DataFrame | PITExpression,
    ) -> PITExpression:
        if isinstance(exponent, PITExpression):
            exponent = exponent.value
        if isinstance(exponent, pd.DataFrame):
            value = np.sign(data.value) * data.value.abs().pow(exponent)
            return PITExpression.from_value(value, data.adj_factor)

        single = data.single_term
        if single is None:
            value = np.sign(data.value) * data.value.abs().pow(exponent)
            return PITExpression.from_value(value, data.adj_factor)
        degree, base = single
        value = np.sign(base) * base.abs().pow(exponent)
        return PITExpression({degree * exponent: value}, data.adj_factor)


PIT_OP = PITOperators()


@contextmanager
def use_universe_mask(mask: pd.DataFrame | None) -> Iterator[None]:
    token = _UNIVERSE_MASK.set(
        None if mask is None else mask.fillna(False).astype(bool)
    )
    try:
        yield
    finally:
        _UNIVERSE_MASK.reset(token)


class PITFactorData:
    """供原Alpha函数读取的轻量点时数据容器。"""

    def __init__(self, fields: Mapping[str, object]) -> None:
        self.fields = dict(fields)

    def require(self, *names: str) -> tuple[object, ...]:
        missing = [name for name in names if name not in self.fields]
        if missing:
            raise ValueError(f"PITFactorData缺少字段：{missing}")
        return tuple(self.fields[name] for name in names)


def build_pit_factor_data(
    cumulative_data: FactorData,
    adj_factor: pd.DataFrame,
) -> PITFactorData:
    """把FactorData转换为带复权次数的数据表达式。"""
    fields: dict[str, object] = {}
    for name, panel in cumulative_data.as_dict().items():
        if name in {"sector", "industry", "subindustry"}:
            fields[name] = panel
        else:
            degree = 1.0 if name in PRICE_FIELDS else 0.0
            fields[name] = PITExpression.from_value(
                panel,
                adj_factor,
                degree=degree,
            )
    return PITFactorData(fields)


def _pit_comparison(
    left: PITExpression | float,
    right: PITExpression | float,
    operator: str,
) -> PITExpression:
    reference = left if isinstance(left, PITExpression) else right
    if not isinstance(reference, PITExpression):
        raise TypeError("比较至少需要一个PITExpression")
    left_expr = reference._coerce(left)
    right_expr = reference._coerce(right)
    operations = {
        "lt": left_expr.value < right_expr.value,
        "le": left_expr.value <= right_expr.value,
        "eq": left_expr.value == right_expr.value,
    }
    result = operations[operator].astype(float)
    result = result.mask(left_expr.value.isna() | right_expr.value.isna())
    return PITExpression.from_value(result, reference.adj_factor)


def _pit_less_than(left: object, right: object) -> PITExpression:
    return _pit_comparison(left, right, "lt")  # type: ignore[arg-type]


def _pit_less_equal(left: object, right: object) -> PITExpression:
    return _pit_comparison(left, right, "le")  # type: ignore[arg-type]


def _pit_equal(left: object, right: object) -> PITExpression:
    return _pit_comparison(left, right, "eq")  # type: ignore[arg-type]


def _pit_logical_or(left: PITExpression, right: PITExpression) -> PITExpression:
    left_value = left.value
    right_value = right.value
    result = pd.DataFrame(np.nan, index=left.index, columns=left.columns)
    result[(left_value == 1) | (right_value == 1)] = 1.0
    result[(left_value == 0) & (right_value == 0)] = 0.0
    return PITExpression.from_value(result, left.adj_factor)


def _pit_as_panel(
    value: PITExpression | float,
    reference: PITExpression,
) -> PITExpression:
    return reference._coerce(value)


def _pit_conditional(
    condition: PITExpression,
    value_if_true: PITExpression | float,
    value_if_false: PITExpression | float,
) -> PITExpression:
    true_value = condition._coerce(value_if_true).value
    false_value = condition._coerce(value_if_false).value
    condition_value = condition.value
    result = false_value.where(condition_value == 0)
    result = result.where(condition_value != 1, true_value)
    return PITExpression.from_value(result, condition.adj_factor)


def _clone_function(function: object, globals_dict: dict[str, object]) -> object:
    if not isinstance(function, FunctionType):
        return function
    cloned = FunctionType(
        function.__code__,
        globals_dict,
        name=function.__name__,
        argdefs=function.__defaults__,
        closure=function.__closure__,
    )
    cloned.__kwdefaults__ = function.__kwdefaults__
    return cloned


def build_point_in_time_functions() -> dict[str, object]:
    """克隆原Alpha函数，并把底层算子替换为点时复权版本。"""
    globals_dict: dict[str, object] = dict(alphas.__dict__)
    globals_dict.update(
        {
            "op": PIT_OP,
            "_comparison": _pit_comparison,
            "_less_than": _pit_less_than,
            "_less_equal": _pit_less_equal,
            "_equal": _pit_equal,
            "_logical_or": _pit_logical_or,
            "_as_panel": _pit_as_panel,
            "_conditional": _pit_conditional,
        }
    )
    for helper_name in ("_adv", "_safe_divide", "_alpha046_trend"):
        globals_dict[helper_name] = _clone_function(
            getattr(alphas, helper_name),
            globals_dict,
        )
    return {
        f"alpha{alpha_id:03d}": _clone_function(
            getattr(alphas, f"alpha{alpha_id:03d}"),
            globals_dict,
        )
        for alpha_id in range(1, 102)
    }


def calculate_point_in_time_factors(
    cumulative_data: FactorData,
    adj_factor: pd.DataFrame,
    *,
    factor_names: list[str] | tuple[str, ...] | None = None,
    universe_mask: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """一次性计算指定因子的逐日点时前复权值。"""
    names = (
        tuple(f"alpha{alpha_id:03d}" for alpha_id in range(1, 102))
        if factor_names is None
        else tuple(factor_names)
    )
    functions = build_point_in_time_functions()
    unknown = set(names).difference(functions)
    if unknown:
        raise ValueError(f"未知Alpha函数：{sorted(unknown)}")
    data = build_pit_factor_data(cumulative_data, adj_factor)
    results: dict[str, pd.DataFrame] = {}
    with use_universe_mask(universe_mask):
        for name in names:
            expression = functions[name](data)
            if isinstance(expression, PITExpression):
                result = expression.value
            elif isinstance(expression, pd.DataFrame):
                result = expression
            else:
                raise TypeError(f"{name}返回了不支持的类型：{type(expression).__name__}")
            result = result.replace([np.inf, -np.inf], np.nan)
            if universe_mask is not None:
                result = result.where(universe_mask)
            result.index.name = "date"
            results[name] = result
    return results
