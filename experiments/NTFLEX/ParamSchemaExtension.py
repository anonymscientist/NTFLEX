from typing import List, Set, Tuple, Optional, Union

from inspect import signature

from TFLEX.expression.symbol import Procedure

NamedSample = List[Tuple[str, int]]

def is_entity(name) -> bool:
    return name.startswith("e") or name.startswith("s") or name.startswith("o")

def is_relation(name) -> bool:
    return name.startswith("r")

def is_timestamp(name) -> bool:
    return name.startswith("t")

def is_value(name) -> bool:
    return name.startswith("x")

def is_attribute(name) -> bool:
    return name.startswith("a")


class QuerySet:

    def __init__(self, ids=None):
        self.ids = ids if ids is not None else set()

    def __len__(self):
        ids_len = len(self.ids) if self.ids is not None else 0
        return ids_len

    def __repr__(self):
        return f"{self.__class__.__name__}({self.ids.__repr__()})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.ids.__repr__()})"

    def __contains__(self, b):
        if self.__class__.__name__ != b.__class__.__name__:
            return False
        if not isinstance(b, QuerySet):
            return False
        return self.ids.issuperset(b.ids)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, QuerySet):
            return False
        if self.__class__.__name__ != __value.__class__.__name__:
            return False
        return self.ids == __value.ids

    def __ne__(self, __value: object) -> bool:
        if not isinstance(__value, QuerySet):
            return True
        if self.__class__.__name__ != __value.__class__.__name__:
            return True
        return self.ids != __value.ids

    def __add__(self, b):
        if self.__class__.__name__ != b.__class__.__name__:
            raise TypeError(f"unsupported operand type(s) for +: '{self.__class__.__name__}' and '{b.__class__.__name__}'")
        ids = self.ids | b.ids
        return self.__class__(ids)

    def __minus__(self, b):
        if self.__class__.__name__ != b.__class__.__name__:
            raise TypeError(f"unsupported operand type(s) for -: '{self.__class__.__name__}' and '{b.__class__.__name__}'")
        ids = self.ids - b.ids
        return self.__class__(ids)

    def __and__(self, b):
        if self.__class__.__name__ != b.__class__.__name__:
            raise TypeError(f"unsupported operand type(s) for &: '{self.__class__.__name__}' and '{b.__class__.__name__}'")
        ids = self.ids & b.ids
        return self.__class__(ids)

    def __or__(self, b):
        if self.__class__.__name__ != b.__class__.__name__:
            raise TypeError(f"unsupported operand type(s) for |: '{self.__class__.__name__}' and '{b.__class__.__name__}'")
        ids = self.ids | b.ids
        return self.__class__(ids)

    def __xor__(self, b):
        if self.__class__.__name__ != b.__class__.__name__:
            raise TypeError(f"unsupported operand type(s) for ^: '{self.__class__.__name__}' and '{b.__class__.__name__}'")
        ids = self.ids ^ b.ids
        return self.__class__(ids)


class EntitySet(QuerySet):
    def __init__(self, entity: Union[int, Set]) -> None:
        if entity is int:
            entity = {entity}
        super().__init__(entity)


class TimeSet(QuerySet):
    def __init__(self, timestamp: Union[int, Set]) -> None:
        if timestamp is int:
            timestamp = {timestamp}
        super().__init__(timestamp)


class AttributeSet(QuerySet):
    def __init__(self, attribute: Union[int, Set]) -> None:
        if attribute is int:
            attribute = {attribute}
        super().__init__(attribute)


class ValueSet(QuerySet):
    def __init__(self, value: Union[int, Set]) -> None:
        if value is int:
            value = {value}
        super().__init__(value)

class MyPlaceholder:

    def __init__(self, name):
        self.name = name
        self.idx: Optional[int] = None

    def __repr__(self):
        return f"MyPlaceholder({self.name}, idx={self.idx})"

    def clear(self):
        self.idx = None

    def fill(self, idx: int):
        self.idx = idx

    def fill_to_fixed_query(self, idx: int):
        self.idx = idx
        return self.to_fixed_query()

    def from_tuple(self, t: Tuple[str, int]):
        type_of_idx, idx = t
        self.name = type_of_idx
        self.idx = idx

    def to_tuple(self) -> Tuple[str, int]:
        return self.name, self.idx
    '''
    def to_fixed_query(self) -> QuerySet:
        
        if is_timestamp(self.name):
            return TimeSet({self.idx})
        elif is_entity(self.name):
            return EntitySet({self.idx})
        elif is_attribute(self.name):
            return AttributeSet({self.idx})
        elif is_value(self.name):
            return ValueSet({self.idx})
        else:
            return QuerySet({self.idx})'
    '''
    '''
    def to_fixed_query(self) -> QuerySet:
        if is_timestamp(self.name):
            return TimeSet({self.idx})
        elif is_entity(self.name):
            return EntitySet({self.idx})
        else:
            return ValueSet({self.idx})'
    '''

    def to_fixed_query(self) -> QuerySet:
        if is_timestamp(self.name):
            return TimeSet({self.idx})
        elif is_value(self.name):
            return ValueSet({self.idx})
        elif is_attribute(self.name):
            return AttributeSet({self.idx})
        else:
            return EntitySet({self.idx})

    def fill_to(self, fixed_query: QuerySet):
        fixed_query.ids = {self.idx}


def get_param_name_list(func) -> List[str]:
    if isinstance(func, Procedure):
        return func.argnames
    sig_func = signature(func)
    return list(sig_func.parameters.keys())


def get_my_placeholder_list(func) -> List[MyPlaceholder]:
    params = get_param_name_list(func)
    return [MyPlaceholder(name) for name in params]


def clear_my_placeholder_list(my_placeholder_list: List[MyPlaceholder]):
    for my_placeholder in my_placeholder_list:
        my_placeholder.clear()


def my_placeholder_to_fixed_query(my_placeholder_list: List[MyPlaceholder], fixed_query_list: List[QuerySet]):
    for my_placeholder, fixed_query in zip(my_placeholder_list, fixed_query_list):
        my_placeholder.fill_to(fixed_query)


def my_placeholder2sample(my_placeholder_list: List[MyPlaceholder]) -> List[int]:
    return [i.idx for i in my_placeholder_list]


def my_placeholder2fixed(my_placeholder_list: List[MyPlaceholder]) -> List[QuerySet]:
    return [i.to_fixed_query() for i in my_placeholder_list]


def sample2namedSample(func, sample: List[int]) -> NamedSample:
    params = get_param_name_list(func)
    return [(name, sample_id) for name, sample_id in zip(params, sample)]
