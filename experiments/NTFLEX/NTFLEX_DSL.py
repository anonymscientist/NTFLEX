"""
@date: 2022/2/21
@description: null
"""
import random
from math import pi
from typing import List, Union

from TFLEX.expression.symbol import Interpreter
from ParamSchemaExtension import MyPlaceholder, EntitySet, QuerySet, TimeSet, AttributeSet, ValueSet, my_placeholder2fixed, get_param_name_list

query_structures = {
    # 10. attribute
    # "PAe": "def PAe(x, a, t): return PAe(x, a, t)",
    # "PAx": "def PAx(e, a, t): return PAx(e, a, t)",
    # "PAt": "def PAt(e, a, x): return PAt(e, a, x)",
    # "PRe": "def PRe(e, r, t): return PRe(e, r, t)",
    # "PRt": "def PRt(e1, r, e2): return PRt(e1, r, e2)",
    "PRe2": "def PRe2(e1, r1, t1, r2, t2): return PRe(PRe(e1, r1, t1), r2, t2)",  # 2p
    #"PRe3": "def PRe3(e1, r1, t1, r2, t2, r3, t3): return PRe(PRe(PRe(e1, r1, t1), r2, t2), r3, t3)",  # 3p
    #"PRe_PAe": "def PRe_PAe(x1, a1, t1, r1, t2): return PRe(PAe(x1, a1, t1), r1, t2)", # 2p
    #"PRe_PRt": "def PRe_PRt(e1, r1, e2, r2, e3): return PRe(e1, r1, PRt(e2, r2, e3))", # 2p
    #"PAe_PAx": "def PAe_PAx(e1, a1, t1, a2, t2): return PAe(PAx(e1, a1, t1), a2, t2)", # 2p
    #"PAe_PAt": "def PAe_PAt(x1, a1, e1, a2, x2): return PAe(x1, a1, PAt(e1, a2, x2))", # 2p
    #"PAx_PAe": "def PAx_PAe(x1, a1, t1, a2, t2): return PAx(PAe(x1, a1, t1), a2, t2)", # 2p
    #"PAx_PAt": "def PAx_PAt(e1, a1, e2, a2, x1): return PAx(e1, a1, PAt(e2, a2, x1))", # 2p
    "PRta": "def PRta(e1, r1, e2): return after(PRt(e1, r1, e2))",  # a for after
    "PRtb": "def PRtb(e1, r1, e2): return before(PRt(e1, r1, e2))",  # b for before
    "ea2i": "def ea2i(x1, a1, t1, x2, a2, t2): return And(PAe(x1, a1, t1), PAe(x2, a2, t2))",  # 2i
    #"ea3i": "def ea3i(x1, a1, t1, x2, a2, t2, x3, a3, t3): return And3(PAe(x1, a1, t1), PAe(x2, a2, t2), PAe(x3, a3, t3))",  # 3i
    "ea2u": "def ea2u(x1, a1, t1, x2, a2, t2): return Or(PAe(x1, a1, t1), PAe(x2, a2, t2))",  # 2u
    "er2i": "def er2i(e1, r1, t1, e2, r2, t2): return And(PRe(e1, r1, t1), PRe(e2, r2, t2))",  # 2i
    #"er3i": "def er3i(e1, r1, t1, e2, r2, t2, e3, r3, t3): return And3(PRe(e1, r1, t1), PRe(e2, r2, t2), PRe(e3, r3, t3))",  # 3i
    "er2u": "def er2u(e1, r1, t1, e2, r2, t2): return Or(PRe(e1, r1, t1), PRe(e2, r2, t2))",  # 2u
    "tr2i": "def tr2i(e1, r1, e2, e3, r2, e4): return TimeAnd(PRt(e1, r1, e2), PRt(e3, r2, e4))",  # t-2i
    #"tr3i": "def tr3i(e1, r1, e2, e3, r2, e4, e5, r3, e6): return TimeAnd3(PRt(e1, r1, e2), PRt(e3, r2, e4), PRt(e5, r3, e6))",  # t-3i
    "tr2u": "def tr2u(e1, r1, e2, e3, r2, e4): return TimeOr(PRt(e1, r1, e2), PRt(e3, r2, e4))",  # t-2u
    "ta2i": "def ta2i(e1, a1, x1, e2, a2, x2): return TimeAnd(PAt(e1, a1, x1), PAt(e2, a2, x2))",  # t-2i
    #"ta3i": "def ta3i(e1, a1, x1, e2, a2, x2, e3, a3, x3): return TimeAnd3(PAt(e1, a1, x1), PAt(e2, a2, x2), PAt(e3, a3, x3))",  # t-3i
    "ta2u": "def ta2u(e1, a1, x1, e2, a2, x2): return TimeOr(PAt(e1, a1, x1), PAt(e2, a2, x2))",  # t-2u
    #"er2i_PRe": "def er2i_PRe(e1, r1, t1, r2, t2, e2, r3, t3): return And(PRe(PRe(e1, r1, t1), r2, t2), PRe(e2, r3, t3))",  # pi
    #"PRe_er2i": "def PRe_er2i(e1, r1, t1, e2, r2, t2, r3, t3): return PRe(er2i(e1, r1, t1, e2, r2, t2), r3, t3)",  # ip
    #"ea2i_PAe": "def ea2i_PAe(e1, a1, t1, a2, t2, x2, a3, t3): return And(PAe(PAx(e1, a1, t1), a2, t2), PAe(x2, a3, t3))",  # pi
    #"PAx_ea2i": "def PAx_ea2i(x1, a1, t1, x2, a2, t2, a3, t3): return PAx(ea2i(x1, a1, t1, x2, a2, t2), a3, t3)",  # ip
    #"tr2i_PRe": "def tr2i_PRe(e1, r1, t1, r2, e2, e3, r3, e4): return TimeAnd(PRt(PRe(e1, r1, t1), r2, e2), PRt(e3, r3, e4))",  # t-pi
    #"PRe_tr2i": "def PRe_tr2i(e1, r1, e2, r2, e3, e4, r3, e5): return PRe(e1, r1, tr2i(e2, r2, e3, e4, r3, e5))",  # t-ip
    #"ta2i_PAe": "def ta2i_PAe(x1, a1, t1, a2, x2, e1, a3, x3): return TimeAnd(PAt(PAe(x1, a1, t1), a2, x2), PAt(e1, a3, x3))",  # t-pi
    #"PAe_ta2i": "def PAe_ta2i(x1, a1, e1, a2, x2, e2, a3, x3): return PAe(x1, a1, ta2i(e1, a2, x2, e2, a3, x3))",  # t-ip
    "gPAx": "def gPAx(e, a, t): return greater(PAx(e, a, t), a)", 
    "sPAx": "def sPAx(e, a, t): return smaller(PAx(e, a, t), a)",
    "gPAxi": "def gPAxi(e1, a1, t1, e2, a2, t2): return ValueAnd(greater(PAx(e1, a1, t1), a1), greater(PAx(e2, a2, t2), a2))",
    "gPAxu": "def gPAxu(e1, a1, t1, e2, a2, t2): return ValueOr(greater(PAx(e1, a1, t1), a1), greater(PAx(e2, a2, t2), a2))",
    "sPAxi": "def sPAxi(e1, a1, t1, e2, a2, t2): return ValueAnd(smaller(PAx(e1, a1, t1), a1), smaller(PAx(e2, a2, t2), a2))",
    "sPAxu": "def sPAxu(e1, a1, t1, e2, a2, t2): return ValueOr(smaller(PAx(e1, a1, t1), a1), smaller(PAx(e2, a2, t2), a2))",
    "gsPAxi": "def gsPAxi(e1, a1, t1, e2, a2, t2): return ValueAnd(greater(PAx(e1, a1, t1), a1), smaller(PAx(e2, a2, t2), a2))",
    "gsPAxu": "def gsPAxu(e1, a1, t1, e2, a2, t2): return ValueOr(greater(PAx(e1, a1, t1), a1), smaller(PAx(e2, a2, t2), a2))",
    "ea2u_DNF": "def ea2u_DNF(x1, a1, t1, x2, a2, t2): return PAe(x1, a1, t1), PAe(x2, a2, t2)",
    "er2u_DNF": "def er2u_DNF(e1, r1, t1, e2, r2, t2): return PRe(e1, r1, t1), PRe(e2, r2, t2)",
    "ta2u_DNF": "def ta2u_DNF(e1, a1, x1, e2, a2, x2): return PAt(e1, a1, x1), PAt(e2, a2, x2)",
    "tr2u_DNF": "def tr2u_DNF(e1, r1, e2, e3, r2, e4): return PRt(e1, r1, e2), PRt(e3, r2, e4)",
    "gPAxu_DNF": "def gPAxu_DNF(e1, a1, t1, e2, a2, t2): return greater(PAx(e1, a1, t1), a1), greater(PAx(e2, a2, t2), a2)",
    "sPAxu_DNF": "def sPAxu_DNF(e1, a1, t1, e2, a2, t2): return smaller(PAx(e1, a1, t1), a1), smaller(PAx(e2, a2, t2), a2)",
    "gsPAxu_DNF": "def gsPAxu_DNF(e1, a1, t1, e2, a2, t2): return greater(PAx(e1, a1, t1), a1), smaller(PAx(e2, a2, t2), a2)",
}

union_query_structures = [
    "ea2u", "er2u",
    "ta2u", "tr2u",
    "gPAxu", "sPAxu", "gsPAxu" # 2u, up
]
train_query_structures = [
    #entity
    #"PAe", "ea2i", "ea3i", "er2i", "er3i", "PRe", "PRe2", "PRe3", "PRe_PAe", "PRe_PRt", "PAe_PAx", "PAe_PAt", "er2i_PRe", 
    #"PRe_er2i", "ea2i_PAe", "PAe_ea2i", "tr2i_PRe", "PRe_tr2i", "ta2i_PAe", "PAe_ta2i",
    "PAe", "PRe", "PRe2", "ea2i", "er2i",
    # value
    #"PAx", "PAx_PAe", "PAx_PAt", "PAxe", "PAxta", "PAxtb", "gPAx", "sPAx", "gPAxi", "sPAxi", "gsPAxi", 
    "PAt", "PRt", "PRta", "PRtb", "ta2i", "tr2i",
    "PAx", "gPAx", "sPAx", "gPAxi", "sPAxi", "gsPAxi",
    # time
    #"PAt", "PAtxg", "PAtxs", "PAter", "PAtea", "PRt", "PRta", "PRtb", "tr2i", "ta2i", "tr3i", "ta3i",
]

test_query_structures = train_query_structures + ["ea2u", "er2u", "ta2u", "tr2u", "gPAxu", "sPAxu", "gsPAxu"]


def is_to_predict_entity_set(query_name) -> bool:
    return query_name.startswith("e") or query_name.startswith("PRe") or query_name.startswith("PAe")

def is_to_predict_timestamp_set(query_name) -> bool:
    return query_name.startswith("t") or query_name.startswith("PRt") or query_name.startswith("PAt")

def contains_relation(query_name) -> bool:
    return query_name in ["PRe", "PRe2", "PRt", "PRta", "PRtb", "er2i", "er2u", "tr2i", "tr2u", "PRe_PRt", "er3i", "tr3i", "er2i_PRe", "PRe_er2i", "tr2i_PRe", "PRe_tr2i"] 

def query_contains_union_and_we_should_use_DNF(query_name) -> bool:
    return query_name in union_query_structures

def filtered_metrics_needed(query_name) -> bool:
    return query_name in ["gPAx", "sPAx", "gPAxi", "sPAxi", "gsPAxi", "PRta", "PRtb", "gPAxu", "sPAxu", "gsPAxu"]


class BasicParser(Interpreter):
    """
    abstract class
    """

    def __init__(self, variables, neural_ops):
        alias = {
            "PAx": neural_ops["AttributeValueProjection"],
            "PAe": neural_ops["AttributeReverseProjection"],
            "PAt": neural_ops["AttributeTimeProjection"],
            "PRe": neural_ops["EntityProjection"],
            "PRt": neural_ops["TimeProjection"],
            "smaller": neural_ops["ValueSmaller"],
            "greater": neural_ops["ValueGreater"],
            "before": neural_ops["TimeBefore"],
            "after": neural_ops["TimeAfter"],
            "next": neural_ops["TimeNext"],
        }
        predefine = {
            "pi": pi,
        }
        functions = dict(**neural_ops, **alias, **predefine)
        super().__init__(usersyms=dict(**variables, **functions))
        self.func_cache = {}
        for _, qs in query_structures.items():
            self.eval(qs)

    def fast_function(self, func_name):
        if func_name in self.func_cache:
            return self.func_cache[func_name]
        func = self.eval(func_name)
        self.func_cache[func_name] = func
        return func

    def fast_args(self, func_name) -> List[str]:
        return get_param_name_list(self.fast_function(func_name))


class SamplingParser(BasicParser):
    def __init__(self, entity_ids: List[int], relations_ids: List[int], attributes_ids: List[int], value_ids: List[int], timestamp_ids: List[int],
                 sax_t, sxa_t, sat_x, sta_x, xas_t, xat_s, xta_s, tas_x, tax_s, ast_x, axt_s, asx_t, t_sax, s_xat, x_sat,
                 sro_t, sor_t, srt_o, str_o, ors_t, trs_o, tro_s, rst_o, rso_t, t_sro, o_srt):
        # example
        # qe = Pe(e,a,after(Pt(e,a,e)))
        # [eid,rid,eid,rid,eid] = qe(e,a,e,a,e)
        # answers = qe(eid,rid,eid,rid,eid)
        # embedding = qe(eid,rid,eid,rid,eid)
        all_entity_ids = set(entity_ids)
        all_timestamp_ids = set(timestamp_ids)
        max_timestamp_id = max(timestamp_ids)
        all_value_ids = set(value_ids)

        variables = {
            "s": MyPlaceholder("s"),
            "t": MyPlaceholder("t"),
            "a": MyPlaceholder("a"),
            "x": MyPlaceholder("x"),
            "e": MyPlaceholder("e"),
            "r": MyPlaceholder("r"),
        }
        for e_id in entity_ids:
            variables[f"e{e_id}"] = EntitySet({e_id})
        for t_id in timestamp_ids:
            variables[f"t{t_id}"] = TimeSet({t_id})
        for a_id in attributes_ids:
            variables[f"a{a_id}"] = AttributeSet({a_id})
        for x_id in value_ids:
            variables[f"x{x_id}"] = ValueSet({x_id})
        for r_id in relations_ids:
            variables[f"r{r_id}"] = QuerySet({r_id})

        def neighbor(s, k=10):
            return random.sample(o_srt[s], k)
        
        def find_value(s: Union[EntitySet, MyPlaceholder], a: Union[QuerySet, MyPlaceholder], t: Union[TimeSet, MyPlaceholder]):
            s_is_missing, a_is_missing, t_is_missing = isinstance(s, MyPlaceholder), isinstance(a, MyPlaceholder), isinstance(t, MyPlaceholder)
            if s_is_missing and a_is_missing and t_is_missing:
                si = random.choice(list(sat_x.keys()))
                s = s.fill_to_fixed_query(si)

                aj = random.choice(list(sat_x[si].keys()))
                a = a.fill_to_fixed_query(aj)

                tk = random.choice(list(sat_x[si][aj].keys()))
                t = t.fill_to_fixed_query(tk)
            elif not s_is_missing and a_is_missing and t_is_missing:
                choices = list(s.ids)
                if len(choices) <= 0:
                    return set()
                si = random.choice(choices)

                choices = list(sat_x[si].keys())
                if len(choices) <= 0:
                    return set()
                aj = random.choice(choices)
                a = a.fill_to_fixed_query(aj)

                choices = list(sat_x[si][aj].keys())
                if len(choices) <= 0:
                    return set()
                tk = random.choice(choices)
                t = t.fill_to_fixed_query(tk)
            elif s_is_missing and not a_is_missing and t_is_missing:
                choices = list(a.ids)
                if len(choices) <= 0:
                    return set()
                aj = random.choice(choices)

                choices = list(ast_x[aj].keys())
                if len(choices) <= 0:
                    return set()
                si = random.choice(choices)
                s = s.fill_to_fixed_query(si)

                choices = list(ast_x[aj][si].keys())
                if len(choices) <= 0:
                    return set()
                tk = random.choice(choices)
                t = t.fill_to_fixed_query(tk)
            elif s_is_missing and a_is_missing and not t_is_missing:
                choices = list(t.ids)
                if len(choices) <= 0:
                    return set()
                tk = random.choice(choices)

                choices = list(tas_x[tk].keys())
                if len(choices) <= 0:
                    return set()
                aj = random.choice(choices)
                a = a.fill_to_fixed_query(aj)

                choices = list(tas_x[tk][aj].keys())
                if len(choices) <= 0:
                    return set()
                si = random.choice(choices)
                s = s.fill_to_fixed_query(si)
            elif s_is_missing and not a_is_missing and not t_is_missing:
                choices = list(t.ids)
                if len(choices) <= 0:
                    return set()
                tk = random.choice(choices)

                choices = list(a.ids)
                if len(choices) <= 0:
                    return set()
                aj = random.choice(choices)

                choices = list(tas_x[tk][aj].keys())
                if len(choices) <= 0:
                    return set()
                si = random.choice(choices)
                s = s.fill_to_fixed_query(si)
            elif not s_is_missing and a_is_missing and not t_is_missing:
                choices = list(s.ids)
                if len(choices) <= 0:
                    return set()
                si = random.choice(choices)

                choices = list(t.ids)
                if len(choices) <= 0:
                    return set()
                tk = random.choice(choices)

                choices = list(sta_x[si][tk].keys())
                if len(choices) <= 0:
                    return set()
                aj = random.choice(choices)
                a = a.fill_to_fixed_query(aj)
            elif not s_is_missing and not a_is_missing and t_is_missing:
                choices = list(s.ids)
                if len(choices) <= 0:
                    return set()
                si = random.choice(choices)

                choices = list(a.ids)
                if len(choices) <= 0:
                    return set()
                aj = random.choice(choices)

                choices = list(sat_x[si][aj].keys())
                if len(choices) <= 0:
                    return set()
                tk = random.choice(choices)
                t = t.fill_to_fixed_query(tk)

            answers = set()
            for si in s.ids:
                for aj in a.ids:
                    for tk in t.ids:
                        answers = answers | set(sat_x[si][aj][tk])
            # print("find_entity", answers)
            return answers

        def find_entity_att(x: Union[ValueSet, MyPlaceholder], a: Union[QuerySet, MyPlaceholder], t: Union[TimeSet, MyPlaceholder]):
            x_is_missing, a_is_missing, t_is_missing = isinstance(x, MyPlaceholder), isinstance(a, MyPlaceholder), isinstance(t, MyPlaceholder)
            if x_is_missing and a_is_missing and t_is_missing:
                xi = random.choice(list(xat_s.keys()))
                x = x.fill_to_fixed_query(xi)

                aj = random.choice(list(xat_s[xi].keys()))
                a = a.fill_to_fixed_query(aj)

                tk = random.choice(list(xat_s[xi][aj].keys()))
                t = t.fill_to_fixed_query(tk)
            elif not x_is_missing and a_is_missing and t_is_missing:
                choices = list(x.ids)
                if len(choices) <= 0:
                    return set()
                xi = random.choice(choices)

                choices = list(xat_s[xi].keys())
                if len(choices) <= 0:
                    return set()
                aj = random.choice(choices)
                a = a.fill_to_fixed_query(aj)

                choices = list(xat_s[xi][aj].keys())
                if len(choices) <= 0:
                    return set()
                tk = random.choice(choices)
                t = t.fill_to_fixed_query(tk)
            elif x_is_missing and not a_is_missing and t_is_missing:
                choices = list(a.ids)
                if len(choices) <= 0:
                    return set()
                aj = random.choice(choices)

                choices = list(axt_s[aj].keys())
                if len(choices) <= 0:
                    return set()
                xi = random.choice(choices)
                x = x.fill_to_fixed_query(xi)

                choices = list(axt_s[aj][xi].keys())
                if len(choices) <= 0:
                    return set()
                tk = random.choice(choices)
                t = t.fill_to_fixed_query(tk)
            elif x_is_missing and a_is_missing and not t_is_missing:
                choices = list(t.ids)
                if len(choices) <= 0:
                    return set()
                tk = random.choice(choices)

                choices = list(tax_s[tk].keys())
                if len(choices) <= 0:
                    return set()
                aj = random.choice(choices)
                a = a.fill_to_fixed_query(aj)

                choices = list(tax_s[tk][aj].keys())
                if len(choices) <= 0:
                    return set()
                xi = random.choice(choices)
                x = x.fill_to_fixed_query(xi)
            elif x_is_missing and not a_is_missing and not t_is_missing:
                choices = list(t.ids)
                if len(choices) <= 0:
                    return set()
                tk = random.choice(choices)

                choices = list(a.ids)
                if len(choices) <= 0:
                    return set()
                aj = random.choice(choices)

                choices = list(tax_s[tk][aj].keys())
                if len(choices) <= 0:
                    return set()
                xi = random.choice(choices)
                x = x.fill_to_fixed_query(xi)
            elif not x_is_missing and a_is_missing and not t_is_missing:
                choices = list(x.ids)
                if len(choices) <= 0:
                    return set()
                xi = random.choice(choices)

                choices = list(t.ids)
                if len(choices) <= 0:
                    return set()
                tk = random.choice(choices)

                choices = list(xta_s[xi][tk].keys())
                if len(choices) <= 0:
                    return set()
                aj = random.choice(choices)
                a = a.fill_to_fixed_query(aj)
            elif not x_is_missing and not a_is_missing and t_is_missing:
                choices = list(x.ids)
                if len(choices) <= 0:
                    return set()
                xi = random.choice(choices)

                choices = list(a.ids)
                if len(choices) <= 0:
                    return set()
                aj = random.choice(choices)

                choices = list(xat_s[xi][aj].keys())
                if len(choices) <= 0:
                    return set()
                tk = random.choice(choices)
                t = t.fill_to_fixed_query(tk)

            answers = set()
            for xi in x.ids:
                for aj in a.ids:
                    for tk in t.ids:
                        answers = answers | set(xat_s[xi][aj][tk])
            # print("find_entity", answers)
            return answers

        def find_timestamp_att(s: Union[EntitySet, MyPlaceholder], a: Union[QuerySet, MyPlaceholder], x: Union[EntitySet, MyPlaceholder]):
            s_is_missing, a_is_missing, x_is_missing = isinstance(s, MyPlaceholder), isinstance(a, MyPlaceholder), isinstance(x, MyPlaceholder)
            if s_is_missing and a_is_missing and x_is_missing:
                si = random.choice(list(sax_t.keys()))
                s = s.fill_to_fixed_query(si)

                aj = random.choice(list(sax_t[si].keys()))
                a = a.fill_to_fixed_query(aj)

                xk = random.choice(list(sax_t[si][aj].keys()))
                x = x.fill_to_fixed_query(xk)
            elif not s_is_missing and a_is_missing and x_is_missing:
                choices = list(s.ids)
                if len(choices) <= 0:
                    return set()
                si = random.choice(choices)

                choices = list(sax_t[si].keys())
                if len(choices) <= 0:
                    return set()
                aj = random.choice(choices)
                a = a.fill_to_fixed_query(aj)

                choices = list(sax_t[si][aj].keys())
                if len(choices) <= 0:
                    return set()
                xk = random.choice(choices)
                x = x.fill_to_fixed_query(xk)
            elif s_is_missing and not a_is_missing and x_is_missing:
                choices = list(a.ids)
                if len(choices) <= 0:
                    return set()
                aj = random.choice(choices)

                choices = list(asx_t[aj].keys())
                if len(choices) <= 0:
                    return set()
                si = random.choice(choices)
                s = s.fill_to_fixed_query(si)

                choices = list(asx_t[aj][si].keys())
                if len(choices) <= 0:
                    return set()
                xk = random.choice(choices)
                x = x.fill_to_fixed_query(xk)
            elif s_is_missing and a_is_missing and not x_is_missing:
                choices = list(x.ids)
                if len(choices) <= 0:
                    return set()
                xk = random.choice(choices)

                choices = list(xas_t[xk].keys())
                if len(choices) <= 0:
                    return set()
                aj = random.choice(choices)
                a = a.fill_to_fixed_query(aj)

                choices = list(xas_t[xk][aj].keys())
                if len(choices) <= 0:
                    return set()
                si = random.choice(choices)
                s = s.fill_to_fixed_query(si)
            elif s_is_missing and not a_is_missing and not x_is_missing:
                choices = list(x.ids)
                if len(choices) <= 0:
                    return set()
                xk = random.choice(choices)

                choices = list(a.ids)
                if len(choices) <= 0:
                    return set()
                aj = random.choice(choices)

                choices = list(tax_s[xk][aj].keys())
                if len(choices) <= 0:
                    return set()
                si = random.choice(choices)
                s = s.fill_to_fixed_query(si)
            elif not s_is_missing and a_is_missing and not x_is_missing:
                choices = list(s.ids)
                if len(choices) <= 0:
                    return set()
                si = random.choice(choices)

                choices = list(x.ids)
                if len(choices) <= 0:
                    return set()
                xk = random.choice(choices)

                choices = list(sxa_t[si][xk].keys())
                if len(choices) <= 0:
                    return set()
                aj = random.choice(choices)
                a = a.fill_to_fixed_query(aj)
            elif not s_is_missing and not a_is_missing and x_is_missing:
                choices = list(s.ids)
                if len(choices) <= 0:
                    return set()
                si = random.choice(choices)

                choices = list(a.ids)
                if len(choices) <= 0:
                    return set()
                aj = random.choice(choices)

                choices = list(sax_t[si][aj].keys())
                if len(choices) <= 0:
                    return set()
                xk = random.choice(choices)
                x = x.fill_to_fixed_query(xk)

            timestamps = set()
            for si in s.ids:
                for aj in a.ids:
                    for xk in x.ids:
                        timestamps = timestamps | set(sax_t[si][aj][xk])
            # print("find_timestamp", timestamps)
            return timestamps
        
        def find_entity_rel(s: Union[EntitySet, MyPlaceholder], r: Union[QuerySet, MyPlaceholder], t: Union[TimeSet, MyPlaceholder]):
            s_is_missing, r_is_missing, t_is_missing = isinstance(s, MyPlaceholder), isinstance(r, MyPlaceholder), isinstance(t, MyPlaceholder)
            if s_is_missing and r_is_missing and t_is_missing:
                si = random.choice(list(srt_o.keys()))
                s = s.fill_to_fixed_query(si)

                rj = random.choice(list(srt_o[si].keys()))
                r = r.fill_to_fixed_query(rj)

                tk = random.choice(list(srt_o[si][rj].keys()))
                t = t.fill_to_fixed_query(tk)
            elif not s_is_missing and r_is_missing and t_is_missing:
                choices = list(s.ids)
                if len(choices) <= 0:
                    return set()
                si = random.choice(choices)

                choices = list(srt_o[si].keys())
                if len(choices) <= 0:
                    return set()
                rj = random.choice(choices)
                r = r.fill_to_fixed_query(rj)

                choices = list(srt_o[si][rj].keys())
                if len(choices) <= 0:
                    return set()
                tk = random.choice(choices)
                t = t.fill_to_fixed_query(tk)
            elif s_is_missing and not r_is_missing and t_is_missing:
                choices = list(r.ids)
                if len(choices) <= 0:
                    return set()
                rj = random.choice(choices)

                choices = list(rst_o[rj].keys())
                if len(choices) <= 0:
                    return set()
                si = random.choice(choices)
                s = s.fill_to_fixed_query(si)

                choices = list(rst_o[rj][si].keys())
                if len(choices) <= 0:
                    return set()
                tk = random.choice(choices)
                t = t.fill_to_fixed_query(tk)
            elif s_is_missing and r_is_missing and not t_is_missing:
                choices = list(t.ids)
                if len(choices) <= 0:
                    return set()
                tk = random.choice(choices)

                choices = list(trs_o[tk].keys())
                if len(choices) <= 0:
                    return set()
                rj = random.choice(choices)
                r = r.fill_to_fixed_query(rj)

                choices = list(trs_o[tk][rj].keys())
                if len(choices) <= 0:
                    return set()
                si = random.choice(choices)
                s = s.fill_to_fixed_query(si)
            elif s_is_missing and not r_is_missing and not t_is_missing:
                choices = list(t.ids)
                if len(choices) <= 0:
                    return set()
                tk = random.choice(choices)

                choices = list(r.ids)
                if len(choices) <= 0:
                    return set()
                rj = random.choice(choices)

                choices = list(trs_o[tk][rj].keys())
                if len(choices) <= 0:
                    return set()
                si = random.choice(choices)
                s = s.fill_to_fixed_query(si)
            elif not s_is_missing and r_is_missing and not t_is_missing:
                choices = list(s.ids)
                if len(choices) <= 0:
                    return set()
                si = random.choice(choices)

                choices = list(t.ids)
                if len(choices) <= 0:
                    return set()
                tk = random.choice(choices)

                choices = list(str_o[si][tk].keys())
                if len(choices) <= 0:
                    return set()
                rj = random.choice(choices)
                r = r.fill_to_fixed_query(rj)
            elif not s_is_missing and not r_is_missing and t_is_missing:
                choices = list(s.ids)
                if len(choices) <= 0:
                    return set()
                si = random.choice(choices)

                choices = list(r.ids)
                if len(choices) <= 0:
                    return set()
                rj = random.choice(choices)

                choices = list(srt_o[si][rj].keys())
                if len(choices) <= 0:
                    return set()
                tk = random.choice(choices)
                t = t.fill_to_fixed_query(tk)

            answers = set()
            for si in s.ids:
                for rj in r.ids:
                    for tk in t.ids:
                        answers = answers | set(srt_o[si][rj][tk])
            # print("find_entity", answers)
            return answers

        def find_timestamp_rel(s: Union[EntitySet, MyPlaceholder], r: Union[QuerySet, MyPlaceholder], o: Union[EntitySet, MyPlaceholder]):
            s_is_missing, r_is_missing, o_is_missing = isinstance(s, MyPlaceholder), isinstance(r, MyPlaceholder), isinstance(o, MyPlaceholder)
            if s_is_missing and r_is_missing and o_is_missing:
                si = random.choice(list(sro_t.keys()))
                s = s.fill_to_fixed_query(si)

                rj = random.choice(list(sro_t[si].keys()))
                r = r.fill_to_fixed_query(rj)

                ok = random.choice(list(sro_t[si][rj].keys()))
                o = o.fill_to_fixed_query(ok)
            elif not s_is_missing and r_is_missing and o_is_missing:
                choices = list(s.ids)
                if len(choices) <= 0:
                    return set()
                si = random.choice(choices)

                choices = list(sro_t[si].keys())
                if len(choices) <= 0:
                    return set()
                rj = random.choice(choices)
                r = r.fill_to_fixed_query(rj)

                choices = list(sro_t[si][rj].keys())
                if len(choices) <= 0:
                    return set()
                ok = random.choice(choices)
                o = o.fill_to_fixed_query(ok)
            elif s_is_missing and not r_is_missing and o_is_missing:
                choices = list(r.ids)
                if len(choices) <= 0:
                    return set()
                rj = random.choice(choices)

                choices = list(rso_t[rj].keys())
                if len(choices) <= 0:
                    return set()
                si = random.choice(choices)
                s = s.fill_to_fixed_query(si)

                choices = list(rso_t[rj][si].keys())
                if len(choices) <= 0:
                    return set()
                ok = random.choice(choices)
                o = o.fill_to_fixed_query(ok)
            elif s_is_missing and r_is_missing and not o_is_missing:
                choices = list(o.ids)
                if len(choices) <= 0:
                    return set()
                ok = random.choice(choices)

                choices = list(ors_t[ok].keys())
                if len(choices) <= 0:
                    return set()
                rj = random.choice(choices)
                r = r.fill_to_fixed_query(rj)

                choices = list(ors_t[ok][rj].keys())
                if len(choices) <= 0:
                    return set()
                si = random.choice(choices)
                s = s.fill_to_fixed_query(si)
            elif s_is_missing and not r_is_missing and not o_is_missing:
                choices = list(o.ids)
                if len(choices) <= 0:
                    return set()
                ok = random.choice(choices)

                choices = list(r.ids)
                if len(choices) <= 0:
                    return set()
                rj = random.choice(choices)

                choices = list(tro_s[ok][rj].keys())
                if len(choices) <= 0:
                    return set()
                si = random.choice(choices)
                s = s.fill_to_fixed_query(si)
            elif not s_is_missing and r_is_missing and not o_is_missing:
                choices = list(s.ids)
                if len(choices) <= 0:
                    return set()
                si = random.choice(choices)

                choices = list(o.ids)
                if len(choices) <= 0:
                    return set()
                ok = random.choice(choices)

                choices = list(sor_t[si][ok].keys())
                if len(choices) <= 0:
                    return set()
                rj = random.choice(choices)
                r = r.fill_to_fixed_query(rj)
            elif not s_is_missing and not r_is_missing and o_is_missing:
                choices = list(s.ids)
                if len(choices) <= 0:
                    return set()
                si = random.choice(choices)

                choices = list(r.ids)
                if len(choices) <= 0:
                    return set()
                rj = random.choice(choices)

                choices = list(sro_t[si][rj].keys())
                if len(choices) <= 0:
                    return set()
                ok = random.choice(choices)
                o = o.fill_to_fixed_query(ok)

            timestamps = set()
            for si in s.ids:
                for rj in r.ids:
                    for ok in o.ids:
                        timestamps = timestamps | set(sro_t[si][rj][ok])
            # print("find_timestamp", timestamps)
            return timestamps
        
        def find_greater_value(x: Union[ValueSet, MyPlaceholder], a: Union[QuerySet, MyPlaceholder]):
            x_is_missing, a_is_missing = isinstance(x, MyPlaceholder), isinstance(a, MyPlaceholder)
            greater = set()
            if x_is_missing and a_is_missing:
                ai = random.choice(list(axt_s.keys()))
                xj = random.choice(list(axt_s[ai].keys()))
                
            elif not x_is_missing and a_is_missing:
                ai = random.choice(list(axt_s.keys()))
                xj = min(x.ids)
                
            elif x_is_missing and not a_is_missing:
                choices = list(a.ids)
                if len(choices) <= 0:
                    return set()
                ai = random.choice(choices)
                xj = random.choice(list(axt_s[ai].keys()))
                
            elif not x_is_missing and not a_is_missing:
                choices = list(a.ids)
                if len(choices) <= 0:
                    return set()
                ai = random.choice(choices)
                xj = min(x.ids)

            for xk in axt_s[ai].keys():
                    if xk > xj:
                        greater.add(xk)
            return greater
        
        def find_smaller_value(x: Union[ValueSet, MyPlaceholder], a: Union[QuerySet, MyPlaceholder]):
            x_is_missing, a_is_missing = isinstance(x, MyPlaceholder), isinstance(a, MyPlaceholder)
            smaller = set()
            if x_is_missing and a_is_missing:
                ai = random.choice(list(axt_s.keys()))
                xj = random.choice(list(axt_s[ai].keys()))
                
            elif not x_is_missing and a_is_missing:
                ai = random.choice(list(axt_s.keys()))
                xj = min(x.ids)
                
            elif x_is_missing and not a_is_missing:
                choices = list(a.ids)
                if len(choices) <= 0:
                    return set()
                ai = random.choice(choices)
                xj = random.choice(list(axt_s[ai].keys()))
                
            elif not x_is_missing and not a_is_missing:
                choices = list(a.ids)
                if len(choices) <= 0:
                    return set()
                ai = random.choice(choices)
                xj = min(x.ids)

            for xk in axt_s[ai].keys():
                    if xk < xj:
                        smaller.add(xk)
            return smaller
                

        def And(q1, q2): return EntitySet(q1.ids & q2.ids)
        def And3(q1, q2, q3): return EntitySet(q1.ids & q2.ids & q3.ids)
        def Or(q1, q2): return EntitySet(q1.ids | q2.ids)
        def AttributeValueProjection(s, a, t): return ValueSet(find_value(s, a, t))
        def AttributeReverseProjection(x, a, t): return EntitySet(find_entity_att(x, a, t))
        def AttributeTimeProjection(s, a, x): return TimeSet(find_timestamp_att(s, a, x))
        def ValueAnd(n1, n2): return ValueSet(n1.ids & n2.ids)
        def ValueOr(n1, n2): return ValueSet(n1.ids | n2.ids)
        def ValueSmaller(n, a): return ValueSet(find_smaller_value(n, a))
        def ValueGreater(n, a): return ValueSet(find_greater_value(n, a))
        def EntityProjection(s, r, t): return EntitySet(find_entity_rel(s, r, t))
        def TimeProjection(s, r, o): return TimeSet(find_timestamp_rel(s, r, o))
        def TimeAnd(q1, q2): return TimeSet(q1.ids & q2.ids)
        def TimeAnd3(q1, q2, q3): return TimeSet(q1.ids & q2.ids & q3.ids)
        def TimeOr(q1, q2): return TimeSet(q1.ids | q2.ids)
        def TimeBefore(x): return TimeSet(set([t for t in all_timestamp_ids if t < min(x.ids)] if len(x.ids) > 0 else all_timestamp_ids))
        def TimeAfter(x): return TimeSet(set([t for t in all_timestamp_ids if t > max(x.ids)] if len(x.ids) > 0 else all_timestamp_ids))
        def TimeNext(x): return TimeSet(set([min(t + 1, max_timestamp_id) for t in x.ids] if len(x.ids) > 0 else all_timestamp_ids))
        neural_ops = {  # 4+4+3
            "And": And,
            "And3": And3,
            "Or": Or,
            "AttributeValueProjection": AttributeValueProjection,
            "AttributeReverseProjection": AttributeReverseProjection,
            "AttributeTimeProjection": AttributeTimeProjection,
            "EntityProjection": EntityProjection,
            "TimeProjection": TimeProjection,
            "ValueAnd": ValueAnd,
            "ValueOr": ValueOr,
            "ValueSmaller": ValueSmaller,
            "ValueGreater": ValueGreater,
            "TimeAnd": TimeAnd,
            "TimeAnd3": TimeAnd3,
            "TimeOr": TimeOr,
            "TimeBefore": TimeBefore,
            "TimeAfter": TimeAfter,
            "TimeNext": TimeNext,
            "neighbor": neighbor,
        }

        valid_ea2i_e_list = [k for k, v in s_xat.items() if len(v) >= 2]

        def fast_ea2i_targeted(x1, a1, t1, x2, a2, t2, target: int):
            (x1_idx, a1_idx, t1_idx), (x2_idx, a2_idx, t2_idx) = tuple(random.sample(list(s_xat[target]), k=2))
            x1.fill(x1_idx)
            a1.fill(a1_idx)
            t1.fill(t1_idx)
            x2.fill(x2_idx)
            a2.fill(a2_idx)
            t2.fill(t2_idx)
            placeholders = [x1, a1, t1, x2, a2, t2]
            return self.fast_function("ea2i")(*my_placeholder2fixed(placeholders))

        def fast_ea2i(x1, a1, t1, x2, a2, t2):
            e_idx = random.choice(valid_ea2i_e_list)
            return fast_ea2i_targeted(x1, a1, t1, x2, a2, t2, target=e_idx)
        
        valid_ea3i_x_list = [k for k, v in s_xat.items() if len(v) >= 3]

        def fast_ea3i(x1, a1, t1, x2, a2, t2, x3, a3, t3):
            x_idx = random.choice(valid_ea3i_x_list)
            (x1_idx, a1_idx, t1_idx), (x2_idx, a2_idx, t2_idx), (x3_idx, a3_idx, t3_idx) = tuple(random.sample(list(s_xat[x_idx]), k=3))
            x1.fill(x1_idx)
            a1.fill(a1_idx)
            t1.fill(t1_idx)
            x2.fill(x2_idx)
            a2.fill(a2_idx)
            t2.fill(t2_idx)
            x3.fill(x3_idx)
            a3.fill(a3_idx)
            t3.fill(t3_idx)
            placeholders = [x1, a1, t1, x2, a2, t2, x3, a3, t3]
            return self.fast_function("ea3i")(*my_placeholder2fixed(placeholders))
        
        valid_er2i_o_list = [k for k, v in o_srt.items() if len(v) >= 2]

        def fast_er2i_targeted(e1, r1, t1, e2, r2, t2, target: int):
            (e1_idx, r1_idx, t1_idx), (e2_idx, r2_idx, t2_idx) = tuple(random.sample(list(o_srt[target]), k=2))
            e1.fill(e1_idx)
            r1.fill(r1_idx)
            t1.fill(t1_idx)
            e2.fill(e2_idx)
            r2.fill(r2_idx)
            t2.fill(t2_idx)
            placeholders = [e1, r1, t1, e2, r2, t2]
            return self.fast_function("er2i")(*my_placeholder2fixed(placeholders))

        def fast_er2i(e1, r1, t1, e2, r2, t2):
            o_idx = random.choice(valid_er2i_o_list)
            return fast_er2i_targeted(e1, r1, t1, e2, r2, t2, target=o_idx)
        
        valid_er3i_o_list = [k for k, v in o_srt.items() if len(v) >= 3]

        def fast_er3i(e1, r1, t1, e2, r2, t2, e3, r3, t3):
            o = random.choice(valid_er3i_o_list)
            (e1_idx, r1_idx, t1_idx), (e2_idx, r2_idx, t2_idx), (e3_idx, r3_idx, t3_idx) = tuple(random.sample(list(o_srt[o]), k=3))
            e1.fill(e1_idx)
            r1.fill(r1_idx)
            t1.fill(t1_idx)
            e2.fill(e2_idx)
            r2.fill(r2_idx)
            t2.fill(t2_idx)
            e3.fill(e3_idx)
            r3.fill(r3_idx)
            t3.fill(t3_idx)
            placeholders = [e1, r1, t1, e2, r2, t2, e3, r3, t3]
            return self.fast_function("er3i")(*my_placeholder2fixed(placeholders))
        
        valid_tr2i_t_list = [k for k, v in t_sro.items() if len(v) >= 2]

        def fast_tr2i(e1, r1, e2, e3, r2, e4):
            t_idx = random.choice(valid_tr2i_t_list)
            return fast_tr2i_targeted(e1, r1, e2, e3, r2, e4, target=t_idx)
        
        def fast_tr2i_targeted(e1, r1, e2, e3, r2, e4, target):
            (e1_idx, r1_idx, e2_idx), (e3_idx, r2_idx, e4_idx) = tuple(random.sample(list(t_sro[target]), k=2))
            e1.fill(e1_idx)
            r1.fill(r1_idx)
            e2.fill(e2_idx)
            e3.fill(e3_idx)
            r2.fill(r2_idx)
            e4.fill(e4_idx)
            placeholders = [e1, r1, e2, e3, r2, e4]
            return self.fast_function("tr2i")(*my_placeholder2fixed(placeholders))
        
        valid_tr3i_t_list = [k for k, v in t_sro.items() if len(v) >= 3]

        def fast_tr3i(e1, r1, e2, e3, r2, e4, e5, r3, e6):
            t = random.choice(valid_tr3i_t_list)
            (e1_idx, r1_idx, e2_idx), (e3_idx, r2_idx, e4_idx), (e5_idx, r3_idx, e6_idx) = tuple(random.sample(list(t_sro[t]), k=3))
            e1.fill(e1_idx)
            r1.fill(r1_idx)
            e2.fill(e2_idx)
            e3.fill(e3_idx)
            r2.fill(r2_idx)
            e4.fill(e4_idx)
            e5.fill(e5_idx)
            r3.fill(r3_idx)
            e6.fill(e6_idx)
            placeholders = [e1, r1, e2, e3, r2, e4, e5, r3, e6]
            return self.fast_function("tr3i")(*my_placeholder2fixed(placeholders))
        
        valid_ta2i_t_list = [k for k, v in t_sax.items() if len(v) >= 2]

        def fast_ta2i_targeted(e1, a1, x1, e2, a2, x2, target):
            (e1_idx, a1_idx, x1_idx), (e2_idx, a2_idx, x2_idx) = tuple(random.sample(list(t_sax[target]), k=2))
            e1.fill(e1_idx)
            a1.fill(a1_idx)
            x1.fill(x1_idx)
            e2.fill(e2_idx)
            a2.fill(a2_idx)
            x2.fill(x2_idx)
            placeholders = [e1, a1, x1, e2, a2, x2]
            return self.fast_function("ta2i")(*my_placeholder2fixed(placeholders))
        
        def fast_ta2i(e1, a1, x1, e2, a2, x2):
            t_idx = random.choice(valid_ta2i_t_list)
            return fast_ta2i_targeted(e1, a1, x1, e2, a2, x2, t_idx)
        
        valid_ta3i_t_list = [k for k, v in t_sax.items() if len(v) >= 3]

        def fast_ta3i(e1, a1, x1, e2, a2, x2, e3, a3, x3):
            t = random.choice(valid_ta3i_t_list)
            (e1_idx, a1_idx, x1_idx), (e2_idx, a2_idx, x2_idx), (e3_idx, a3_idx, x3_idx) = tuple(random.sample(list(t_sax[t]), k=3))
            e1.fill(e1_idx)
            a1.fill(a1_idx)
            x1.fill(x1_idx)
            e2.fill(e2_idx)
            a2.fill(a2_idx)
            x2.fill(x2_idx)
            e3.fill(e3_idx)
            a3.fill(a3_idx)
            x3.fill(x3_idx)
            placeholders = [e1, a1, x1, e2, a2, x2, e3, a3, x3]
            return self.fast_function("ta3i")(*my_placeholder2fixed(placeholders))
        
        def fast_PRe_targeted(e1, r1, t1, target: int):
            e1_idx, r1_idx, t1_idx = random.choice(list(o_srt[target]))
            e1.fill(e1_idx)
            r1.fill(r1_idx)
            t1.fill(t1_idx)
            return srt_o[e1_idx][r1_idx][t1_idx]
        
        def fast_PRe2_targeted(e1, r1, t1, r2, t2, target: int):
            # PRe(PRe(e1, r1, t1), r2, t2)
            e1_idx, r2_idx, t2_idx = random.choice(list(o_srt[target]))
            e1_ids = fast_PRe_targeted(e1, r1, t1, target=e1_idx)
            r2.fill(r2_idx)
            t2.fill(t2_idx)
            answers = set()
            for idx in e1_ids:
                answers = answers | srt_o[idx][r2_idx][t2_idx]
            return answers

        def fast_PRt_targeted(e1, r1, e2, target: int):
            e1_idx, r1_idx, e2_idx = random.choice(list(t_sro[target]))
            e1.fill(e1_idx)
            r1.fill(r1_idx)
            e2.fill(e2_idx)
            return sro_t[e1_idx][r1_idx][e2_idx]
        
        def fast_PAt_targeted(e1, a2, x2, target: int):
            e1_idx, a2_idx, x2_idx = random.choice(list(t_sax[target]))
            e1.fill(e1_idx)
            a2.fill(a2_idx)
            x2.fill(x2_idx)
            return sax_t[e1_idx][a2_idx][x2_idx]
        
        def fast_PAx_targeted(e1, a1, t1, target: int):
            e1_idx, a1_idx, t1_idx = random.choice(list(x_sat[target]))
            e1.fill(e1_idx)
            a1.fill(a1_idx)
            t1.fill(t1_idx)
            return sat_x[e1_idx][a1_idx][t1_idx]
        
        def fast_PAe_targeted(x1, a1, t1, target: int):
            x1_idx, a1_idx, t1_idx = random.choice(list(s_xat[target]))
            x1.fill(x1_idx)
            a1.fill(a1_idx)
            t1.fill(t1_idx)
            return xat_s[x1_idx][a1_idx][t1_idx]
        
        def fast_PRe_PRt(e1, r1, e2, r2, e3):
            # return PRe(e1, r1, PRt(e2, r2, e3))
            o_idx = random.choice(list(o_srt.keys()))
            e1_idx, r1_idx, t1_idx = random.choice(list(o_srt[o_idx]))
            e1.fill(e1_idx)
            r1.fill(r1_idx)
            o_ids = set()
            t_ids = fast_PRt_targeted(e2, r2, e3, target=t1_idx)
            for t_idx in t_ids:
                o_ids = o_ids | srt_o[e1_idx][r1_idx][t_idx]
            return EntitySet(o_ids)
        
        def fast_PRt_PRe_targeted(e1, r1, t1, r2, e2, target):
            # PRt(PRe(e1, r1, t1), r2, e2)
            e1_idx, r2_idx, e2_idx = random.choice(list(t_sro[target]))
            e1_ids = fast_PRe_targeted(e1, r1, t1, target=e1_idx)
            r2.fill(r2_idx)
            e2.fill(e2_idx)
            answers = set()
            for idx in e1_ids:
                answers = answers | sro_t[idx][r2_idx][e2_idx]
            return answers
        
        def fast_PAt_PAe_targeted(x1, a1, t1, a2, x2, target):
            # return PAt(PAe(x1, a1, t1), a2, x2)
            e1_idx, a2_idx, x2_idx = random.choice(list(t_sax[target]))
            e_ids = fast_PAe_targeted(x1, a1, t1, target=e1_idx)
            a2.fill(a2_idx)
            x2.fill(x2_idx)
            answers = set()
            for e_idx in e_ids:
                answers = answers | sax_t[e_idx][a2_idx][x2_idx]
            return answers
        
        def fast_PAe_PAt(x1, a1, e1, a2, x2):
            # return PAe(x1, a1, PAt(e1, a2, x2))
            e_idx = random.choice(list(s_xat.keys()))
            x1_idx, a1_idx, t1_idx = random.choice(list(s_xat[e_idx]))
            x1.fill(x1_idx)
            a1.fill(a1_idx)
            e_ids = set()
            t_ids = fast_PAt_targeted(e1, a2, x2, target=t1_idx)
            for t_idx in t_ids:
                e_ids = e_ids | xat_s[x1_idx][a1_idx][t_idx]
            return EntitySet(e_ids)
        
        def fast_PAe_PAx(e1, a1, t1, a2, t2): 
            #return PAe(PAx(e1, a1, t1), a2, t2)
            e_idx = random.choice(list(s_xat.keys()))
            x1_idx, a2_idx, t2_idx = random.choice(list(s_xat[e_idx]))
            x1_ids = fast_PAx_targeted(e1, a1, t1, target=x1_idx)
            a2.fill(a2_idx)
            t2.fill(t2_idx)
            e_ids = set()
            for idx in x1_ids:
                e_ids = e_ids | xat_s[idx][a2_idx][t2_idx]
            return EntitySet(e_ids)
        
        def fast_PAe_PAx_targeted(e1, a1, t1, a2, t2, target):
            # PAe(PAx(e1, a1, t1), a2, t2)
            x1_idx, a2_idx, t2_idx = random.choice(list(s_xat[target]))
            x1_ids = fast_PAx_targeted(e1, a1, t1, target=x1_idx)
            a2.fill(a2_idx)
            t2.fill(t2_idx)
            answers = set()
            for idx in x1_ids:
                answers = answers | xat_s[idx][a2_idx][t2_idx]
            return answers
        
        def fast_PAx_PAt(e1, a1, e2, a2, x1): 
            # return PAx(e1, a1, PAt(e2, a2, x1))
            x_idx = random.choice(list(x_sat.keys()))
            e1_idx, a1_idx, t1_idx = random.choice(list(x_sat[x_idx]))
            e1.fill(e1_idx)
            a1.fill(a1_idx)
            x_ids = set()
            t_ids = fast_PAt_targeted(e2, a2, x1, target=t1_idx)
            for t_idx in t_ids:
                x_ids = x_ids | sat_x[e1_idx][a1_idx][t_idx]
            return ValueSet(x_ids)
        
        def fast_PAx_PAe(x1, a1, t1, a2, t2): 
            # return PAx(PAe(x1, a1, t1), a2, t2)
            x_idx = random.choice(list(x_sat.keys()))
            e1_idx, a2_idx, t2_idx = random.choice(list(x_sat[x_idx]))
            a2.fill(a2_idx)
            t2.fill(t2_idx)
            x_ids = set()
            e_ids = fast_PAe_targeted(x1, a1, t1, target=e1_idx)
            for e_idx in e_ids:
                x_ids = x_ids | sat_x[e_idx][a2_idx][t2_idx]
            return ValueSet(x_ids)

        def fast_PRe_er2i(e1, r1, t1, e2, r2, t2, r3, t3):
            # return PRe(And(PRe(e1, r1, t1), PRe(e2, r2, t2)), r3, t3)
            o_idx = random.choice(list(set(valid_er2i_o_list) & set(sro_t.keys())))
            q = fast_er2i_targeted(e1, r1, t1, e2, r2, t2, target=o_idx)
            return self.fast_function("PRe")(q, r3, t3)

        def fast_er2i_PRe(e1, r1, t1, r2, t2, e2, r3, t3):
            # return And(PRe(PRe(e1, r1, t1), r2, t2), PRe(e2, r3, t3))
            o_idx = random.choice(valid_er2i_o_list)
            right_o_ids = fast_PRe_targeted(e2, r3, t3, target=o_idx)
            left_o_ids = fast_PRe2_targeted(e1, r1, t1, r2, t2, target=o_idx)
            return EntitySet(left_o_ids & right_o_ids)
        
        def fast_ea2i_PAe(e1, a1, t1, a2, t2, x2, a3, t3): 
            #return And(PAe(PAx(e1, a1, t1), a2, t2), PAe(x2, a3, t3))
            e_idx = random.choice(valid_ea2i_e_list)
            right_e_ids = fast_PAe_targeted(x2, a3, t3, target=e_idx)
            left_e_ids = fast_PAe_PAx_targeted(e1, a1, t1, a2, t2, target=e_idx)
            return EntitySet(left_e_ids & right_e_ids)

        def fast_PAx_ea2i(x1, a1, t1, x2, a2, t2, a3, t3): 
            #return PAx(ea2i(x1, a1, t1, x2, a2, t2), a3, t3)
            e_idx = random.choice(list(set(valid_ea2i_e_list) & set(sax_t.keys())))
            q = fast_ea2i_targeted(x1, a1, t1, x2, a2, t2, target=e_idx)
            return self.fast_function("PAx")(q, a3, t3)

        def fast_tr2i_PRe(e1, r1, t1, r2, e2, e3, r3, e4): 
            #return TimeAnd(PRt(PRe(e1, r1, t1), r2, e2), PRt(e3, r3, e4))
            t_idx = random.choice(valid_tr2i_t_list)
            right_t_ids = fast_PRt_targeted(e3, r3, e4, target=t_idx)
            left_t_ids = fast_PRt_PRe_targeted(e1, r1, t1, r2, e2, target=t_idx)
            return TimeSet(left_t_ids & right_t_ids)

        def fast_PRe_tr2i(e1, r1, e2, r2, e3, e4, r3, e5): 
            #return PRe(e1, r1, tr2i(e2, r2, e3, e4, r3, e5))
            t_idx = random.choice(list(set(valid_tr2i_t_list) & set(tro_s.keys())))
            t = fast_tr2i_targeted(e2, r2, e3, e4, r3, e5, target=t_idx)
            return self.fast_function("PRe")(e1, r1, t)

        def fast_ta2i_PAe(x1, a1, t1, a2, x2, e1, a3, x3): 
            #return TimeAnd(PAt(PAe(x1, a1, t1), a2, x2), PAt(e1, a3, x3))
            t_idx = random.choice(valid_ta2i_t_list)
            right_t_ids = fast_PAt_targeted(e1, a3, x3, target=t_idx)
            left_t_ids = fast_PAt_PAe_targeted(x1, a1, t1, a2, x2, target=t_idx)
            return TimeSet(left_t_ids & right_t_ids)

        def fast_PAe_ta2i(x1, a1, e1, a2, x2, e2, a3, x3): 
            #return PAe(x1, a1, ta2i(e1, a2, x2, e2, a3, x3))
            t_idx = random.choice(list(set(valid_ta2i_t_list) & set(tax_s.keys())))
            t = fast_ta2i_targeted(e1, a2, x2, e2, a3, x3, target=t_idx)
            return self.fast_function("PAe")(x1, a1, t)
        
        self.fast_ops = {
            "fast_er2i": fast_er2i,
            "fast_ea2i": fast_ea2i,
            "fast_tr2i": fast_tr2i,
            "fast_ta2i": fast_ta2i,
            "fast_er3i": fast_er3i,
            "fast_ea3i": fast_ea3i,
            "fast_tr3i": fast_tr3i,
            "fast_ta3i": fast_ta3i,
            "fast_PRe_PRt": fast_PRe_PRt,
            "fast_PAe_PAt": fast_PAe_PAt,
            "fast_PAe_PAx": fast_PAe_PAx,
            "fast_PAx_PAt": fast_PAx_PAt,
            "fast_PAx_PAe": fast_PAx_PAe,
            "fast_PRe_er2i": fast_PRe_er2i,
            "fast_er2i_PRe": fast_er2i_PRe,
            "fast_PAe_ta2i": fast_PAe_ta2i,
            "fast_ta2i_PAe": fast_ta2i_PAe,
            "fast_PRe_tr2i": fast_PRe_tr2i,
            "fast_tr2i_PRe": fast_tr2i_PRe,
            "fast_PAx_ea2i": fast_PAx_ea2i,
            "fast_ea2i_PAe": fast_ea2i_PAe,
        }

        super().__init__(variables=variables, neural_ops=dict(**neural_ops, **self.fast_ops))


class NeuralParser(BasicParser):
    def __init__(self, neural_ops, variables=None):
        if variables is None:
            variables = {}
        must_implement_neural_ops = set({
            #"And",
            #"Or",
            "AttributeValueProjection",
            #"AttributeReverseProjection",
            #"AttributeTimeProjection",
            "EntityProjection",
            #"TimeProjection",
            #"ValueAnd",
            #"ValueOr",
            #"ValueSmaller",
            #"ValueGreater",
            #"TimeAnd",
            #"TimeOr",
            #"TimeBefore",
            #"TimeAfter",
            #"TimeNext",
        })
        ops = set(neural_ops.keys())
        not_implemented_ops = must_implement_neural_ops - ops
        if len(not_implemented_ops) > 0:
            raise Exception(f"You MUST implement neural operation '{not_implemented_ops}'")
        super().__init__(variables=variables, neural_ops=neural_ops)
