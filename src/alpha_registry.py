"""WorldQuant 101 Alphas 的统一注册表。

注册表只保存元数据和函数引用，不在模块导入时读取行情。
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

import alphas
from factor_data import FactorData


AlphaFunction = Callable[[FactorData], pd.DataFrame]
VALID_IMPLEMENTATION_STATUSES = {"implemented", "not_implemented"}
VALID_RUN_STATUSES = {"not_run", "success", "failed", "skipped"}
FRACTIONAL_WINDOW_ALPHA_IDS = {
    58, 59, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
    73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87,
    88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
}
ADV_ALPHA_IDS = {
    7, 17, 21, 25, 28, 31, 36, 39, 43, 47, 61, 62, 63, 64,
    65, 67, 68, 69, 70, 71, 72, 74, 75, 76, 77, 78, 79, 80,
    81, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
    98, 99, 100,
}


@dataclass(slots=True)
class AlphaSpec:
    """单个 Alpha 的实现与数据依赖元数据。"""

    alpha_id: int
    function_name: str
    required_fields: tuple[str, ...]
    industry_levels: tuple[str, ...]
    uses_market_cap: bool
    max_lookback: int
    implementation_batch: int
    status: str
    is_approximate: bool = False
    notes: str = ""
    last_run_status: str = "not_run"
    last_error: str = ""
    function: AlphaFunction | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not 1 <= self.alpha_id <= 101:
            raise ValueError("alpha_id必须介于1至101")
        if self.status not in VALID_IMPLEMENTATION_STATUSES:
            raise ValueError(f"未知实现状态：{self.status}")
        if self.last_run_status not in VALID_RUN_STATUSES:
            raise ValueError(f"未知运行状态：{self.last_run_status}")
        if self.max_lookback < 0:
            raise ValueError("max_lookback不能为负数")

    @property
    def alpha_name(self) -> str:
        return f"alpha{self.alpha_id:03d}"

    @property
    def industry_level(self) -> tuple[str, ...] | None:
        """兼容单数字段名，实际保留所有行业层级。"""
        return self.industry_levels or None

    def record_success(self) -> None:
        self.last_run_status = "success"
        self.last_error = ""

    def record_failure(self, error: BaseException) -> None:
        self.last_run_status = "failed"
        self.last_error = f"{type(error).__name__}: {error}"

    def record_skipped(self, reason: str) -> None:
        self.last_run_status = "skipped"
        self.last_error = reason

    def as_dict(self) -> dict[str, object]:
        """返回可序列化的注册信息，不包含函数对象。"""
        return {
            "alpha_id": self.alpha_id,
            "function_name": self.function_name,
            "required_fields": list(self.required_fields),
            "industry_level": list(self.industry_levels) or None,
            "uses_market_cap": self.uses_market_cap,
            "max_lookback": self.max_lookback,
            "implementation_batch": self.implementation_batch,
            "status": self.status,
            "is_approximate": self.is_approximate,
            "notes": self.notes,
            "last_run_status": self.last_run_status,
            "last_error": self.last_error,
        }


# alpha_id|required_fields|industry_levels|max_lookback|implementation_batch
# max_lookback表示公式中最大的显式窗口，不累加嵌套窗口的预热期。
_METADATA = """
001|close,returns||20|3
002|open,close,volume||6|2
003|open,volume||10|2
004|low||9|2
005|open,close,vwap||10|1
006|open,volume||10|2
007|close,volume||60|3
008|open,returns||10|1
009|close||5|3
010|close||4|3
011|close,volume,vwap||3|1
012|close,volume||1|1
013|close,volume||5|2
014|open,volume,returns||10|2
015|high,volume||3|2
016|high,volume||5|2
017|close,volume||20|2
018|open,close||10|2
019|close,returns||250|1
020|open,high,low,close||1|1
021|close,volume||20|3
022|high,close,volume||20|2
023|high||20|3
024|close||100|3
025|high,close,vwap,returns||20|1
026|high,volume||5|2
027|volume,vwap||6|3
028|high,low,close||20|2
029|close,returns||6|2
030|close,volume||20|1
031|low,close||20|2
032|close,vwap||230|2
033|open,close||0|1
034|close,returns||5|1
035|high,low,close,volume,returns||32|2
036|open,close,volume,vwap,returns||200|2
037|open,close||200|2
038|open,close||10|2
039|close,volume,returns||250|2
040|high,volume||10|2
041|high,low,vwap||0|1
042|close,vwap||0|1
043|close,volume||20|2
044|high,volume||5|2
045|close,volume||20|2
046|close||20|3
047|high,close,volume,vwap||20|1
048|close|subindustry|250|4
049|close||20|3
050|volume,vwap||5|2
051|close||20|3
052|low,volume,returns||240|2
053|high,low,close||9|1
054|open,high,low,close||0|1
055|high,low,close,volume||12|2
056|returns,cap||10|4
057|close,vwap||30|2
058|volume,vwap|sector|8|4
059|volume,vwap|industry|17|4
060|high,low,close,volume||10|1
061|vwap||180|2
062|open,high,low,vwap||23|2
063|open,close,vwap|industry|180|4
064|open,high,low,vwap||120|2
065|open,vwap||60|2
066|open,high,low,vwap||12|2
067|high,vwap|sector,subindustry|20|4
068|high,low,close||15|2
069|close,vwap|industry|20|4
070|close,vwap|industry|50|4
071|open,low,close,vwap||180|2
072|high,low,volume,vwap||40|2
073|open,low,vwap||17|2
074|high,close,volume,vwap||38|2
075|low,volume,vwap||50|2
076|low,vwap|sector|81|4
077|high,low,vwap||40|2
078|low,volume,vwap||40|2
079|open,close,vwap|sector|150|4
080|open,high|industry|10|4
081|volume,vwap||50|2
082|open,volume|sector|18|4
083|high,low,close,volume,vwap||5|1
084|close,vwap||21|2
085|high,low,close,volume||30|2
086|open,close,vwap||21|2
087|close,vwap|industry|81|4
088|open,high,low,close||60|2
089|low,vwap|industry|16|4
090|low,close|subindustry|40|4
091|close,volume,vwap|industry|30|4
092|open,high,low,close||30|2
093|close,vwap|industry|81|4
094|vwap||60|2
095|open,high,low||40|2
096|close,volume,vwap||60|2
097|low,vwap|industry|60|4
098|open,vwap||27|2
099|high,low,volume||60|2
100|high,low,close,volume|subindustry|30|4
101|open,high,low,close||0|1
""".strip()


def _split_values(value: str) -> tuple[str, ...]:
    return tuple(item for item in value.split(",") if item)


def _build_registry() -> dict[int, AlphaSpec]:
    registry: dict[int, AlphaSpec] = {}
    for line in _METADATA.splitlines():
        raw_id, raw_fields, raw_levels, raw_lookback, raw_batch = line.split("|")
        alpha_id = int(raw_id)
        function_name = f"alpha{alpha_id:03d}"
        function = getattr(alphas, function_name, None)
        if function is not None and not callable(function):
            raise TypeError(f"{function_name}存在但不可调用")
        required_fields = _split_values(raw_fields)
        if alpha_id in ADV_ALPHA_IDS:
            required_fields = tuple(dict.fromkeys((*required_fields, "volume", "amount")))
        uses_fractional_windows = alpha_id in FRACTIONAL_WINDOW_ALPHA_IDS
        registry[alpha_id] = AlphaSpec(
            alpha_id=alpha_id,
            function_name=function_name,
            required_fields=required_fields,
            industry_levels=_split_values(raw_levels),
            uses_market_cap="cap" in required_fields,
            max_lookback=int(raw_lookback),
            implementation_batch=int(raw_batch),
            status="implemented" if function is not None else "not_implemented",
            is_approximate=False,
            notes=(
                "formula-v2：小数窗口按论文向下取整"
                if uses_fractional_windows and function is not None
                else ""
            ),
            function=function,
        )

    if sorted(registry) != list(range(1, 102)):
        raise RuntimeError("Alpha注册表未完整覆盖1至101")
    return registry


ALPHA_REGISTRY = _build_registry()


def normalize_alpha_id(value: int | str) -> int:
    """将 12、'12'、'alpha012'或'alpha_012'统一转换为12。"""
    if isinstance(value, bool):
        raise TypeError("alpha_id不能是bool")
    if isinstance(value, int):
        alpha_id = value
    elif isinstance(value, str):
        match = re.fullmatch(r"(?:alpha_?)?(\d{1,3})", value.strip(), flags=re.I)
        if not match:
            raise ValueError(f"无法识别Alpha编号：{value}")
        alpha_id = int(match.group(1))
    else:
        raise TypeError("alpha_id必须是int或str")
    if alpha_id not in ALPHA_REGISTRY:
        raise KeyError(f"Alpha {alpha_id}不在注册表中")
    return alpha_id


def get_alpha_spec(alpha_id: int | str) -> AlphaSpec:
    """按编号或名称返回注册项。"""
    return ALPHA_REGISTRY[normalize_alpha_id(alpha_id)]


def select_alpha_specs(
    alpha_ids: list[int | str] | None = None,
    *,
    implemented_only: bool = False,
) -> list[AlphaSpec]:
    """按注册顺序选择因子，并拒绝重复编号。"""
    if alpha_ids is None:
        specs = list(ALPHA_REGISTRY.values())
    else:
        normalized = [normalize_alpha_id(value) for value in alpha_ids]
        if len(normalized) != len(set(normalized)):
            raise ValueError("Alpha选择列表不能包含重复编号")
        specs = [ALPHA_REGISTRY[alpha_id] for alpha_id in normalized]
    if implemented_only:
        specs = [spec for spec in specs if spec.status == "implemented"]
    return specs


def registry_records() -> list[dict[str, object]]:
    """返回101条可序列化的注册记录。"""
    return [spec.as_dict() for spec in ALPHA_REGISTRY.values()]


def save_registry_manifest(file_path: Path) -> None:
    """将当前注册表状态保存为CSV。"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    records = registry_records()
    columns = list(records[0])
    with file_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    **record,
                    "required_fields": ",".join(record["required_fields"]),
                    "industry_level": ",".join(record["industry_level"] or []),
                }
            )


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    output = project_root / "docs" / "alpha_registry.csv"
    save_registry_manifest(output)
    print(f"已保存101条Alpha注册信息：{output}")
