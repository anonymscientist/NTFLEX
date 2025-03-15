from collections import defaultdict
from pathlib import Path
from pprint import pformat
from typing import List, Tuple, Dict, Set, Union, Any

from ParamSchemaExtension import QuerySet, my_placeholder2sample, get_param_name_list, get_my_placeholder_list, my_placeholder2fixed, clear_my_placeholder_list
from TFLEX.assistance.toolbox.data.DataSchema import DatasetCachePath, BaseData
from TFLEX.assistance.toolbox.data.DatasetSchema import RelationalTripletDatasetSchema
from TFLEX.assistance.toolbox.data.functional import read_cache, cache_data
from TFLEX.assistance.toolbox.utils.Progbar import Progbar

from NTFLEX_DSL import *


class WIKI(RelationalTripletDatasetSchema):
    def __init__(self, home: Union[Path, str] = "data"):
        super(WIKI, self).__init__("WIKI", home)

    def get_data_paths(self) -> Dict[str, Path]:
        return {
            'train': self.get_dataset_path_child('train'),
            'test': self.get_dataset_path_child('test'),
            'valid': self.get_dataset_path_child('valid'),
        }

    def get_dataset_path(self):
        return self.root_path 


class NumericalKnowledgeDatasetCachePath(DatasetCachePath):
    def __init__(self, cache_path: Path):
        DatasetCachePath.__init__(self, cache_path)
        self.cache_all_att_triples_path = self.cache_path / 'triplets_all_att.pkl'
        self.cache_all_rel_triples_path = self.cache_path / 'triplets_all_rel.pkl'
        self.cache_train_att_triples_path = self.cache_path / 'triplets_train_att.pkl'
        self.cache_train_rel_triples_path = self.cache_path / 'triplets_train_rel.pkl'
        self.cache_test_att_triples_path = self.cache_path / 'triplets_test_att.pkl'
        self.cache_test_rel_triples_path = self.cache_path / 'triplets_test_rel.pkl'
        self.cache_valid_att_triples_path = self.cache_path / 'triplets_valid_att.pkl'
        self.cache_valid_rel_triples_path = self.cache_path / 'triplets_valid_rel.pkl'

        self.cache_all_att_triples_ids_path = self.cache_path / 'triplets_ids_all_att.pkl'
        self.cache_all_rel_triples_ids_path = self.cache_path / 'triplets_ids_all_rel.pkl'
        self.cache_train_att_triples_ids_path = self.cache_path / 'triplets_ids_train_att.pkl'
        self.cache_train_rel_triples_ids_path = self.cache_path / 'triplets_ids_train_rel.pkl'
        self.cache_test_att_triples_ids_path = self.cache_path / 'triplets_ids_test_att.pkl'
        self.cache_test_rel_triples_ids_path = self.cache_path / 'triplets_ids_test_rel.pkl'
        self.cache_valid_att_triples_ids_path = self.cache_path / 'triplets_ids_valid_att.pkl'
        self.cache_valid_rel_triples_ids_path = self.cache_path / 'triplets_ids_valid_rel.pkl'

        self.cache_all_entities_path = self.cache_path / 'entities.pkl'
        self.cache_all_attributes_path = self.cache_path / 'attributes.pkl'
        self.cache_all_relations_path = self.cache_path / 'relations.pkl'
        self.cache_all_timestamps_path = self.cache_path / 'timestamps.pkl'
        self.cache_all_values_path = self.cache_path / 'values.pkl'
        self.cache_entities_ids_path = self.cache_path / "entities_ids.pkl"
        self.cache_attributes_ids_path = self.cache_path / "attributes_ids.pkl"
        self.cache_relations_ids_path = self.cache_path / 'relations_ids.pkl'
        self.cache_timestamps_ids_path = self.cache_path / "timestamps_ids.pkl"
        self.cache_values_ids_path = self.cache_path / "values_ids.pkl"

        self.cache_idx2entity_path = self.cache_path / 'idx2entity.pkl'
        self.cache_idx2attribute_path = self.cache_path / 'idx2attribute.pkl'
        self.cache_idx2relation_path = self.cache_path / 'idx2relation.pkl'
        self.cache_idx2timestamp_path = self.cache_path / 'idx2timestamp.pkl'
        self.cache_idx2value_path = self.cache_path / 'idx2value.pkl'
        self.cache_entity2idx_path = self.cache_path / 'entity2idx.pkl'
        self.cache_attribute2idx_path = self.cache_path / 'attribute2idx.pkl'
        self.cache_relation2idx_path = self.cache_path / 'relation2idx.pkl'
        self.cache_timestamp2idx_path = self.cache_path / 'timestamp2idx.pkl'
        self.cache_value2idx_path = self.cache_path / 'value2idx.pkl'


def read_triples_sort_saxt(file_path: Union[str, Path]) -> Tuple[List[Tuple[str, str, str, str]], List[Tuple[str, str, str, str]]]:
    """
    return [(lhs, att, val, timestamp)]
              s    a,   x,       t
    """
    with open(str(file_path), 'r', encoding='utf-8') as fr:
        triple_saxt = set()
        triple_srot = set()
        for line in fr.readlines():
            try:
                lhs, att, val, _, since, _, until = line.strip().split('\t')
                try:
                    if since != "None" and until != "None":
                        start = int(float(since))
                        end = int(float(until))
                        if val[0] == "Q":
                            while start <= end:
                                triple_srot.add((lhs, att, val, str(start)))
                                start += 1
                        else:
                            while start <= end:
                                triple_saxt.add((lhs, att, val, str(start)))
                                start += 1
                    elif since == "None" and until != "None":
                        if val[0] == "Q":
                            triple_srot.add((lhs, att, val, until))
                        else:
                            triple_saxt.add((lhs, att, val, until))
                    elif since != "None" and until == "None":
                        if val[0] == "Q":
                            triple_srot.add((lhs, att, val, since))
                        else:
                            triple_saxt.add((lhs, att, val, since))
                except IndexError as e:
                    print(f"IndexError: {e}, line: {line}")
            except ValueError:
                lhs, att, val, timestamp = line.strip().split('\t')
                try:
                    if val[0] == "Q":
                        triple_srot.add((lhs, att, val, timestamp))
                    else:
                        triple_saxt.add((lhs, att, val, timestamp))
                except IndexError as e:
                    print(f"IndexError: {e}, line: {line}")
    return list(triple_saxt), list(triple_srot)


TYPE_MAPPING_sax_t = Dict[int, Dict[int, Dict[int, Set[int]]]]
TYPE_MAPPING_sat_x = Dict[int, Dict[int, Dict[int, Set[int]]]]
TYPE_MAPPING_t_sax = Dict[int, Set[Tuple[int, int, int]]]
TYPE_MAPPING_x_sat = Dict[int, Set[Tuple[int, int, int]]]


def build_map_t2sax_and_x2sat(triples_ids: List[Tuple[int, int, int, int]]) -> Tuple[TYPE_MAPPING_t_sax, TYPE_MAPPING_x_sat]:
    t_sax = defaultdict(set)
    x_sat = defaultdict(set)
    for s, a, x, t in triples_ids:
        t_sax[t].add((s, a, x))
        x_sat[x].add((s, a, t))
    return t_sax, x_sat


def build_map_sax_t(triplets: List[Tuple[int, int, int, int]]) -> Dict[Tuple[int, int, int], Set[int]]:
    """ Function to read the list of tails for the given head and relation pair. """
    sax_t: Dict[Tuple[int, int, int], Set[int]] = defaultdict(set)
    for s, a, x, t in triplets:
        sax_t[(s, a, x)].add(t)

    return sax_t


def build_map_sat_x(triplets: List[Tuple[int, int, int, int]]) -> Dict[Tuple[int, int, int], Set[int]]:
    """ Function to read the list of tails for the given head and relation pair. """
    sat_x: Dict[Tuple[int, int, int], Set[int]] = defaultdict(set)
    for s, a, x, t in triplets:
        sat_x[(s, a, t)].add(x)

    return sat_x


def build_map_sax2t_and_sat2x(triples_ids: List[Tuple[int, int, int, int]]) -> Tuple[TYPE_MAPPING_sax_t, TYPE_MAPPING_sat_x]:
    """ Function to read the list of tails for the given head and relation pair. """
    sax_t = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    sat_x = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    for s, a, x, t in triples_ids:
        sax_t[s][a][x].add(t)
        sat_x[s][a][t].add(x)
    return sax_t, sat_x

def build_map_sax2t_and_sat2x_and_xat2s(triples_ids: List[Tuple[int, int, int, int]]) -> Tuple[TYPE_MAPPING_sax_t, TYPE_MAPPING_sat_x]:
    """ Function to read the list of tails for the given head and relation pair. """
    sax_t = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    sat_x = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))    
    xat_s = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    for s, a, x, t in triples_ids:
        sax_t[s][a][x].add(t)
        sat_x[s][a][t].add(x)
        xat_s[x][a][t].add(s)
    return sax_t, sat_x, xat_s

def build_map_srt_o(triplets: List[Tuple[int, int, int, int]]) -> Dict[Tuple[int, int, int], Set[int]]:
    """ Function to read the list of tails for the given head and relation pair. """
    srt_o: Dict[Tuple[int, int, int], Set[int]] = defaultdict(set)
    for s, r, o, t in triplets:
        srt_o[(s, r, t)].add(o)

    return srt_o

def build_mapping_simple_att(triples_ids: List[Tuple[int, int, int, int]]):
    sax_t = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    sxa_t = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    sat_x = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    sta_x = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    xas_t = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    xat_s = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    xta_s = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    tas_x = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    tax_s = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    ast_x = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    axt_s = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    asx_t = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    t_sax = defaultdict(set)
    s_xat = defaultdict(set)
    x_sat = defaultdict(set)
    for s, a, x, t in triples_ids:
        sax_t[s][a][x].add(t)
        sxa_t[s][x][a].add(t)
        sat_x[s][a][t].add(x)
        sta_x[s][t][a].add(x)
        xas_t[x][a][s].add(t)
        xat_s[x][a][t].add(s)
        xta_s[x][t][a].add(s)
        tas_x[t][a][s].add(x)
        tax_s[t][a][x].add(s)
        ast_x[a][s][t].add(x)
        axt_s[a][x][t].add(s)
        asx_t[a][s][x].add(t)
        t_sax[t].add((s, a, x))
        s_xat[s].add((x, a, t))
        x_sat[x].add((s, a, t))
    return sax_t, sxa_t, sat_x, sta_x, \
           xas_t, xat_s, xta_s, tas_x, \
           tax_s, ast_x, axt_s, asx_t, \
           t_sax, s_xat, x_sat


def build_mapping_simple_rel(triples_ids: List[Tuple[int, int, int, int]]):
    sro_t = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    sor_t = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    srt_o = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    str_o = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    ors_t = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    trs_o = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    tsr_o = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    tro_s = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    rst_o = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    rso_t = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    t_sro = defaultdict(set)
    o_srt = defaultdict(set)
    for s, r, o, t in triples_ids:
        sro_t[s][r][o].add(t)
        sor_t[s][o][r].add(t)
        srt_o[s][r][t].add(o)
        str_o[s][t][r].add(o)
        ors_t[o][r][s].add(t)
        trs_o[t][r][s].add(o)
        tsr_o[t][s][r].add(o)
        tro_s[t][r][o].add(s)
        rst_o[r][s][t].add(o)
        rso_t[r][s][o].add(t)
        t_sro[t].add((s, r, o))
        o_srt[o].add((s, r, t))
    return sro_t, sor_t, srt_o, str_o, \
           ors_t, trs_o, tro_s, rst_o, \
           rso_t, t_sro, o_srt


def build_not_t2sax_x2sat(entities_ids: List[int], timestamps_ids: List[int],
                          sax_t: TYPE_MAPPING_sax_t, sat_x: TYPE_MAPPING_sat_x) -> Tuple[TYPE_MAPPING_t_sax, TYPE_MAPPING_x_sat]:
    # DON'T USE THIS FUNCTION! THERE ARE DRAGONS!
    not_t_sax = defaultdict(set)
    not_x_sat = defaultdict(set)
    for s in sax_t:
        for a in sax_t[s]:
            for x in sax_t[s][a]:
                for t in set(timestamps_ids) - set(sax_t[s][a][x]):  # negation on timestamps
                    not_t_sax[t].add((s, a, x))
    for s in sat_x:
        for a in sat_x[s]:
            for t in sat_x[s][a]:
                for x in set(entities_ids) - set(sat_x[s][a][t]):  # negation on entities
                    not_x_sat[x].add((s, a, t))
    return not_t_sax, not_x_sat


class NumericalKnowledgeData(BaseData):
    """ The class is the main module that handles the knowledge graph.

        KnowledgeGraph is responsible for downloading, parsing, processing and preparing
        the training, testing and validation dataset.

        Args:
            dataset (RelationalTripletDatasetSchema): custom dataset.
            cache_path (TemporalKnowledgeDatasetCachePath): cache path.

        Attributes:
            dataset (RelationalTripletDatasetSchema): custom dataset.
            cache_path (TemporalKnowledgeDatasetCachePath): cache path.

            all_attributes (list):list of all the relations.
            all_entities (list): List of all the entities.
            all_timestamps (list): List of all the timestamps.

            entity2idx (dict): Dictionary for mapping string name of entities to unique numerical id.
            idx2entity (dict): Dictionary for mapping the entity id to string.
            relation2idx (dict): Dictionary for mapping string name of relations to unique numerical id.
            idx2relation (dict): Dictionary for mapping the relation id to string.
            timestamp2idx (dict): Dictionary for mapping string name of timestamps to unique numerical id.
            idx2timestamp (dict): Dictionary for mapping the timestamp id to string.

        Examples:
            >>> from ComplexTemporalQueryData import ICEWS14, TemporalKnowledgeDatasetCachePath, TemporalKnowledgeData
            >>> dataset = ICEWS14()
            >>> cache = TemporalKnowledgeDatasetCachePath(dataset.cache_path)
            >>> data = TemporalKnowledgeData(dataset=dataset, cache_path=cache)
            >>> data.preprocess_data_if_needed()

    """

    def __init__(self,
                 dataset: RelationalTripletDatasetSchema,
                 cache_path: NumericalKnowledgeDatasetCachePath):
        BaseData.__init__(self, dataset, cache_path)
        self.dataset = dataset
        self.cache_path = cache_path

        # KG data structure stored in triplet format
        self.all_att_triples: List[Tuple[str, str, str, str]] = []  # (s, a, x, t)
        self.all_rel_triples: List[Tuple[str, str, str, str]] = []  # (s, r, o, t)
        self.train_att_triples: List[Tuple[str, str, str, str]] = []
        self.train_rel_triples: List[Tuple[str, str, str, str]] = []
        self.test_att_triples: List[Tuple[str, str, str, str]] = []
        self.test_rel_triples: List[Tuple[str, str, str, str]] = []
        self.valid_att_triples: List[Tuple[str, str, str, str]] = []
        self.valid_rel_triples: List[Tuple[str, str, str, str]] = []

        self.all_att_triples_ids: List[Tuple[int, int, int, int]] = []  # (s, a, x, t)
        self.train_att_triples_ids: List[Tuple[int, int, int, int]] = []
        self.test_att_triples_ids: List[Tuple[int, int, int, int]] = []
        self.valid_att_triples_ids: List[Tuple[int, int, int, int]] = []
        self.all_rel_triples_ids: List[Tuple[int, int, int, int]] = []  # (s, r, o, t)
        self.train_rel_triples_ids: List[Tuple[int, int, int, int]] = []
        self.test_rel_triples_ids: List[Tuple[int, int, int, int]] = []
        self.valid_rel_triples_ids: List[Tuple[int, int, int, int]] = []

        self.all_attributes: List[str] = []  # name
        self.all_relations: List[str] = []
        self.all_entities: List[str] = []
        self.all_att_entities: List[str] = []
        self.all_rel_entities: List[str] = []
        self.all_timestamps: List[str] = []
        self.all_att_timestamps: List[str] = []
        self.all_rel_timestamps: List[str] = []
        self.all_values: List[str] = []
        self.entities_ids: List[int] = []
        self.entities_att_ids: List[int] = []  # id
        self.entities_rel_ids: List[int] = []
        self.attributes_ids: List[int] = []
        self.relations_ids: List[int] = []
        self.timestamps_ids: List[int] = []
        self.timestamps_att_ids: List[int] = []
        self.timestamps_rel_ids: List[int] = []
        self.values_ids: List[int] = []

        self.entity2idx: Dict[str, int] = {}
        self.idx2entity: Dict[int, str] = {}
        self.entity2idx_rel: Dict[str, int] = {}
        self.idx2entity_rel: Dict[int, str] = {}
        self.entity2idx_att: Dict[str, int] = {}
        self.idx2entity_att: Dict[int, str] = {}
        self.attribute2idx: Dict[str, int] = {}
        self.idx2attribute: Dict[int, str] = {}
        self.relation2idx: Dict[str, int] = {}
        self.idx2relation: Dict[int, str] = {}
        self.timestamp2idx: Dict[str, int] = {}
        self.idx2timestamp: Dict[int, str] = {}
        self.timestamp2idx_rel: Dict[str, int] = {}
        self.idx2timestamp_rel: Dict[int, str] = {}
        self.timestamp2idx_att: Dict[str, int] = {}
        self.idx2timestamp_att: Dict[int, str] = {}
        self.value2idx: Dict[int, str] = {}
        self.idx2value: Dict[int, str] = {}

        # meta
        self.entity_count = 0
        self.entity_rel_count = 0
        self.entity_att_count = 0
        self.attribute_count = 0
        self.relation_count = 0
        self.timestamp_count = 0
        self.timestamp_rel_count = 0
        self.timestamp_att_count = 0
        self.value_count = 0
        self.valid_triples_count = 0
        self.test_triples_count = 0
        self.train_triples_count = 0
        self.triple_count = 0

    def read_all_origin_data(self):
        self.read_all_triplets()

    def read_all_triplets(self):
        self.train_att_triples, self.train_rel_triples = read_triples_sort_saxt(self.dataset.data_paths['train'])
        self.valid_att_triples, self.valid_rel_triples = read_triples_sort_saxt(self.dataset.data_paths['valid'])
        self.test_att_triples, self.test_rel_triples = read_triples_sort_saxt(self.dataset.data_paths['test'])
        self.all_att_triples = self.train_att_triples + self.valid_att_triples + self.test_att_triples
        self.all_rel_triples = self.train_rel_triples + self.valid_rel_triples + self.test_rel_triples

        self.valid_triples_count = len(self.valid_att_triples) + len(self.valid_rel_triples)
        self.test_triples_count = len(self.test_att_triples) + len(self.test_rel_triples)
        self.train_triples_count = len(self.train_att_triples) + len(self.train_rel_triples)
        self.triple_count = self.valid_triples_count + self.test_triples_count + self.train_triples_count

    def transform_all_data(self):
        self.transform_ent_rel_att_val_time()
        self.transform_mappings()
        self.transform_all_triplets_ids()

        self.transform_entity_ids()
        self.transform_attribute_ids()
        self.transform_relation_ids()
        self.transform_value_ids()
        self.transform_timestamp_ids()

    def transform_ent_rel_att_val_time(self):
        """ Function to read the entities. """
        entities: Set[str] = set()
        entities_rel: Set[str] = set()
        entities_att: Set[str] = set()
        relations: Set[str] = set()
        timestamps: Set[str] = set()
        timestamps_rel: Set[str] = set()
        timestamps_att: Set[str] = set()
        attributes: Set[str] = set()
        values: Set[str] = set()

        # print("entities_relations")
        # bar = Progbar(len(self.all_triples))
        # i = 0
        for s, r, o, t in self.all_rel_triples:
            entities.add(s)
            entities_rel.add(s)
            entities.add(o)
            entities_rel.add(o)
            relations.add(r)
            timestamps.add(t)
            timestamps_rel.add(t)

        for s, a, x, t in self.all_att_triples:
            entities.add(s)
            entities_att.add(s)
            attributes.add(a)
            values.add(x)
            timestamps.add(t)
            timestamps_att.add(t)
        
        self.all_entities = sorted(list(entities))
        self.all_rel_entities = sorted(list(entities_rel))
        self.all_att_entities = sorted(list(entities_att))
        self.all_relations = sorted(list(relations))
        self.all_attributes = sorted(list(attributes))
        self.all_values = sorted(list(values))
        self.all_timestamps = sorted(list(timestamps))
        self.all_rel_timestamps = sorted(list(timestamps_rel))
        self.all_att_timestamps = sorted(list(timestamps_att))

        self.entity_count = len(self.all_entities)
        self.entity_rel_count = len(self.all_rel_entities)
        self.entity_att_count = len(self.all_att_entities)
        self.relation_count = len(self.all_relations)
        self.attribute_count = len(self.all_attributes)
        self.value_count = len(self.all_values)
        self.timestamp_count = len(self.all_timestamps)
        self.timestamp_rel_count = len(self.all_rel_timestamps)
        self.timestamp_att_count = len(self.all_att_timestamps)

    def transform_mappings(self):
        """ Function to generate the mapping from string name to integer ids. """
        for k, v in enumerate(self.all_entities):
            self.entity2idx[v] = k
            self.idx2entity[k] = v
            if v in self.all_att_entities:
                self.entity2idx_att[v] = k
                self.idx2entity_att[k] = v
            if v in self.all_rel_entities:
                self.entity2idx_rel[v] = k
                self.idx2entity_rel[k] = v
        for k, v in enumerate(self.all_attributes):
            self.attribute2idx[v] = k
            self.idx2attribute[k] = v
        for k, v in enumerate(self.all_relations):
            self.relation2idx[v] = k
            self.idx2relation[k] = v
        for k, v in enumerate(self.all_values):
            self.value2idx[v] = k
            self.idx2value[k] = v
        for k, v in enumerate(self.all_timestamps):
            self.timestamp2idx[v] = k
            self.idx2timestamp[k] = v
            if v in self.all_att_timestamps:
                self.timestamp2idx_att[v] = k
                self.idx2timestamp_att[k] = v
            if v in self.all_rel_timestamps:
                self.timestamp2idx_rel[v] = k
                self.idx2timestamp_rel[k] = v

    def transform_all_triplets_ids(self):
        entity2idx_att = self.entity2idx_att
        entity2idx_rel = self.entity2idx_rel
        attribute2idx = self.attribute2idx
        relation2idx = self.relation2idx
        value2idx = self.value2idx
        timestamp2idx_att = self.timestamp2idx_att
        timestamp2idx_rel = self.timestamp2idx_rel
        self.train_att_triples_ids = [(entity2idx_att[s], attribute2idx[a], value2idx[x], timestamp2idx_att[t]) for s, a, x, t in self.train_att_triples]
        self.test_att_triples_ids = [(entity2idx_att[s], attribute2idx[a], value2idx[x], timestamp2idx_att[t]) for s, a, x, t in self.test_att_triples]
        self.valid_att_triples_ids = [(entity2idx_att[s], attribute2idx[a], value2idx[x], timestamp2idx_att[t]) for s, a, x, t in self.valid_att_triples]
        self.all_att_triples_ids = self.train_att_triples_ids + self.valid_att_triples_ids + self.test_att_triples_ids
        
        self.train_rel_triples_ids = [(entity2idx_rel[s], relation2idx[r], entity2idx_rel[o], timestamp2idx_rel[t]) for s, r, o, t in self.train_rel_triples]
        self.test_rel_triples_ids = [(entity2idx_rel[s], relation2idx[r], entity2idx_rel[o], timestamp2idx_rel[t]) for s, r, o, t in self.test_rel_triples]
        self.valid_rel_triples_ids = [(entity2idx_rel[s], relation2idx[r], entity2idx_rel[o], timestamp2idx_rel[t]) for s, r, o, t in self.valid_rel_triples]
        self.all_rel_triples_ids = self.train_rel_triples_ids + self.valid_rel_triples_ids + self.test_rel_triples_ids

    def transform_entity_ids(self):
        entity2idx = self.entity2idx
        entity2idx_att = self.entity2idx_att
        entity2idx_rel = self.entity2idx_rel
        for e in self.all_entities:
            self.entities_ids.append(entity2idx[e])
        for e in self.all_att_entities:
            self.entities_att_ids.append(entity2idx_att[e])
        for e in self.all_rel_entities:
            self.entities_rel_ids.append(entity2idx_rel[e])
        print("entities_ids", len(self.entities_ids))
        print("entities_att_ids", len(self.entities_att_ids))
        print("entities_rel_ids", len(self.entities_rel_ids))

    def transform_attribute_ids(self):
        attribute2idx = self.attribute2idx
        for r in self.all_attributes:
            self.attributes_ids.append(attribute2idx[r])
        print("attributes_ids", len(self.attributes_ids))

    def transform_relation_ids(self):
        relation2idx = self.relation2idx
        for r in self.all_relations:
            self.relations_ids.append(relation2idx[r])
        print("relations_ids", len(self.relations_ids))

    def transform_value_ids(self):
        value2idx = self.value2idx
        for r in self.all_values:
            self.values_ids.append(value2idx[r])
        print("values_ids", len(self.values_ids))

    def transform_timestamp_ids(self):
        timestamp2idx = self.timestamp2idx
        timestamp2idx_att = self.timestamp2idx_att
        timestamp2idx_rel = self.timestamp2idx_rel
        for t in self.all_timestamps:
            self.timestamps_ids.append(timestamp2idx[t])
        for t in self.all_att_timestamps:
            self.timestamps_att_ids.append(timestamp2idx_att[t])
        for t in self.all_rel_timestamps:
            self.timestamps_rel_ids.append(timestamp2idx_rel[t])
        print("timestamp_ids", len(self.timestamps_ids))
        print("timestamps_att_ids", len(self.timestamps_att_ids))
        print("timestamps_rel_ids", len(self.timestamps_rel_ids))

    def cache_all_data(self):
        """Function to cache the prepared dataset in the memory"""
        cache_data(self.all_att_triples, self.cache_path.cache_all_att_triples_path)
        cache_data(self.all_rel_triples, self.cache_path.cache_all_rel_triples_path)
        cache_data(self.train_att_triples, self.cache_path.cache_train_att_triples_path)
        cache_data(self.train_rel_triples, self.cache_path.cache_train_rel_triples_path)
        cache_data(self.test_att_triples, self.cache_path.cache_test_att_triples_path)
        cache_data(self.test_rel_triples, self.cache_path.cache_test_rel_triples_path)
        cache_data(self.valid_att_triples, self.cache_path.cache_valid_att_triples_path)
        cache_data(self.valid_rel_triples, self.cache_path.cache_valid_rel_triples_path)

        cache_data(self.all_att_triples_ids, self.cache_path.cache_all_att_triples_ids_path)
        cache_data(self.all_rel_triples_ids, self.cache_path.cache_all_rel_triples_ids_path)
        cache_data(self.train_att_triples_ids, self.cache_path.cache_train_att_triples_ids_path)
        cache_data(self.train_rel_triples_ids, self.cache_path.cache_train_rel_triples_ids_path)
        cache_data(self.test_att_triples_ids, self.cache_path.cache_test_att_triples_ids_path)
        cache_data(self.test_rel_triples_ids, self.cache_path.cache_test_rel_triples_ids_path)
        cache_data(self.valid_att_triples_ids, self.cache_path.cache_valid_att_triples_ids_path)
        cache_data(self.valid_rel_triples_ids, self.cache_path.cache_valid_rel_triples_ids_path)

        cache_data(self.all_entities, self.cache_path.cache_all_entities_path)
        cache_data(self.all_attributes, self.cache_path.cache_all_attributes_path)
        cache_data(self.all_relations, self.cache_path.cache_all_relations_path)
        cache_data(self.all_values, self.cache_path.cache_all_values_path)
        cache_data(self.all_timestamps, self.cache_path.cache_all_timestamps_path)
        cache_data(self.entities_ids, self.cache_path.cache_entities_ids_path)
        cache_data(self.attributes_ids, self.cache_path.cache_attributes_ids_path)
        cache_data(self.relations_ids, self.cache_path.cache_relations_ids_path)
        cache_data(self.values_ids, self.cache_path.cache_values_ids_path)
        cache_data(self.timestamps_ids, self.cache_path.cache_timestamps_ids_path)

        cache_data(self.idx2entity, self.cache_path.cache_idx2entity_path)
        cache_data(self.idx2attribute, self.cache_path.cache_idx2attribute_path)
        cache_data(self.idx2relation, self.cache_path.cache_idx2relation_path)
        cache_data(self.idx2value, self.cache_path.cache_idx2value_path)
        cache_data(self.idx2timestamp, self.cache_path.cache_idx2timestamp_path)
        cache_data(self.entity2idx, self.cache_path.cache_entity2idx_path)
        cache_data(self.attribute2idx, self.cache_path.cache_attribute2idx_path)
        cache_data(self.relation2idx, self.cache_path.cache_relation2idx_path)
        cache_data(self.value2idx, self.cache_path.cache_value2idx_path)
        cache_data(self.timestamp2idx, self.cache_path.cache_timestamp2idx_path)

        cache_data(self.meta(), self.cache_path.cache_metadata_path)

    def load_cache(self, keys: List[str]):
        for key in keys:
            self.read_cache_data(key)

    def read_cache_data(self, key):
        """Function to read the cached dataset from the memory"""
        path = "cache_%s_path" % key
        if hasattr(self, key) and hasattr(self.cache_path, path):
            key_path = getattr(self.cache_path, path)
            value = read_cache(key_path)
            setattr(self, key, value)
            return value
        elif key == "meta":
            meta = read_cache(self.cache_path.cache_metadata_path)
            self.read_meta(meta)
        else:
            raise ValueError('Unknown cache data key %s' % key)

    def read_meta(self, meta):
        self.entity_count = meta["entity_count"]
        self.relation_count = meta["relation_count"]
        self.attribute_count = meta["attribute_count"]
        self.value_count = meta["value_count"]
        self.timestamp_count = meta["timestamp_count"]
        self.valid_triples_count = meta["valid_triples_count"]
        self.test_triples_count = meta["test_triples_count"]
        self.train_triples_count = meta["train_triples_count"]
        self.triple_count = meta["triple_count"]

    def meta(self) -> Dict[str, Any]:
        return {
            "entity_count": self.entity_count,
            "relation_count": self.relation_count,
            "attribute_count": self.attribute_count,
            "value_count": self.value_count,
            "timestamp_count": self.timestamp_count,
            "valid_triples_count": self.valid_triples_count,
            "test_triples_count": self.test_triples_count,
            "train_triples_count": self.train_triples_count,
            "triple_count": self.triple_count,
        }

    def dump(self) -> List[str]:
        """ Function to dump statistic information of a dataset """
        # dump key information
        dump = [
            "",
            "-" * 15 + "Metadata Info for Dataset: " + self.dataset.name + "-" * (15 - len(self.dataset.name)),
            "Total Training Triples   :%s" % self.train_triples_count,
            "Total Testing Triples    :%s" % self.test_triples_count,
            "Total validation Triples :%s" % self.valid_triples_count,
            "Total Entities           :%s" % self.entity_count,
            "Total Relations          :%s" % self.relation_count,
            "Total Attributes         :%s" % self.attribute_count,
            "Total Values             :%s" % self.value_count,
            "Total Timestamps         :%s" % self.timestamp_count,
            "-" * (30 + len("Metadata Info for Dataset: ")),
            "",
        ]
        return dump


"""
above is simple temporal kg
below is complex query data (logical reasoning) based on previous temporal kg
"""


class ComplexNumericalQueryDatasetCachePath(NumericalKnowledgeDatasetCachePath):
    def __init__(self, cache_path: Path):
        NumericalKnowledgeDatasetCachePath.__init__(self, cache_path)
        self.cache_train_queries_answers_path = self.cache_path / "train_queries_answers.pkl"
        self.cache_valid_queries_answers_path = self.cache_path / "valid_queries_answers.pkl"
        self.cache_test_queries_answers_path = self.cache_path / "test_queries_answers.pkl"

    def cache_queries_answers_path(self, split: str, query_name: str) -> Path:
        return self.cache_path / f"{split}_{query_name}_queries_answers.pkl"


TYPE_train_queries_answers = Dict[str, Dict[str, Union[List[str], List[Tuple[List[int], Set[int]]]]]]
TYPE_test_queries_answers = Dict[str, Dict[str, Union[List[str], List[Tuple[List[int], Set[int], Set[int]]]]]]


class NumericalComplexQueryData(NumericalKnowledgeData):

    def __init__(self,
                 dataset: RelationalTripletDatasetSchema,
                 cache_path: ComplexNumericalQueryDatasetCachePath):
        NumericalKnowledgeData.__init__(self, dataset, cache_path)
        self.cache_path = cache_path
        # Dict[str, Dict[str, Union[List[str], List[Tuple[List[int], Set[int]]]]]]
        #       |                       |                     |          |
        #     structure name      args name list              |          |
        #                                    ids corresponding to args   |
        #                                                          answers id set
        # 1. `structure name` is the name of a function (named query function), parsed to AST and eval to get results.
        # 2. `args name list` is the arg list of query function.
        # 3. train_queries_answers, valid_queries_answers and test_queries_answers are heavy to load (~10G+ memory)
        #    we suggest to load by query task, e.g. load_cache_by_tasks(["Pe", "Pe2", "Pe3", "e2i", "e3i"], "train")
        self.train_queries_answers: TYPE_train_queries_answers = {
            # "Pe_aPt": {
            #     "args": ["e1", "r1", "e2", "r2", "e3"],
            #     "queries_answers": [
            #         ([1, 2, 3, 4, 5], {2, 3, 5}),
            #         ([1, 2, 3, 4, 5], {2, 3, 5}),
            #         ([1, 2, 3, 4, 5], {2, 3, 5}),
            #     ]
            # }
            # >>> answers = Pe_aPt(1, 2, 3, 4, 5)
            # then, answers == {2, 3}
        }
        self.valid_queries_answers: TYPE_test_queries_answers = {
            # "Pe_aPt": {
            #     "args": ["e1", "r1", "e2", "r2", "e3"],
            #     "queries_answers": [
            #         ([1, 2, 3, 4, 5], {2, 3}, {2, 3, 5}),
            #         ([1, 2, 3, 4, 5], {2, 3}, {2, 3, 5}),
            #         ([1, 2, 3, 4, 5], {2, 3}, {2, 3, 5}),
            #     ]
            # }
            # >>> answers = Pe_aPt(1, 2, 3, 4, 5)
            # in training set, answers == {2, 3}
            # in validation set, answers == {2, 3, 5}, harder and more complete
        }
        self.test_queries_answers: TYPE_test_queries_answers = {
            # "Pe_aPt": {
            #     "args": ["e1", "r1", "e2", "r2", "e3"],
            #     "queries_answers": [
            #         ([1, 2, 3, 4, 5], {2, 3, 5}, {2, 3, 5, 6}),
            #         ([1, 2, 3, 4, 5], {2, 3, 5}, {2, 3, 5, 6}),
            #         ([1, 2, 3, 4, 5], {2, 3, 5}, {2, 3, 5, 6}),
            #     ]
            # }
            # >>> answers = Pe_aPt(1, 2, 3, 4, 5)
            # in training and validation set, answers == {2, 3, 5}
            # in testing set, answers == {2, 3, 5, 6}, harder and more complete
        }
        # meta
        self.query_meta = {
            # "Pe_aPt": {
            #     "queries_count": 1,
            #     "avg_answers_count": 1
            # }
        }

    def restore_from_cache(self):
        self.all_att_triples = read_cache(self.cache_path.cache_all_att_triples_path)
        self.train_att_triples = read_cache(self.cache_path.cache_train_att_triples_path)
        self.test_att_triples = read_cache(self.cache_path.cache_test_att_triples_path)
        self.valid_att_triples = read_cache(self.cache_path.cache_valid_att_triples_path)
        self.all_att_triples_ids = read_cache(self.cache_path.cache_all_att_triples_ids_path)
        self.train_att_triples_ids = read_cache(self.cache_path.cache_train_att_triples_ids_path)
        self.test_att_triples_ids = read_cache(self.cache_path.cache_test_att_triples_ids_path)
        self.valid_att_triples_ids = read_cache(self.cache_path.cache_valid_att_triples_ids_path)
        self.all_rel_triples = read_cache(self.cache_path.cache_all_rel_triples_path)
        self.train_rel_triples = read_cache(self.cache_path.cache_train_rel_triples_path)
        self.test_rel_triples = read_cache(self.cache_path.cache_test_rel_triples_path)
        self.valid_rel_triples = read_cache(self.cache_path.cache_valid_rel_triples_path)
        self.all_rel_triples_ids = read_cache(self.cache_path.cache_all_rel_triples_ids_path)
        self.train_rel_triples_ids = read_cache(self.cache_path.cache_train_rel_triples_ids_path)
        self.test_rel_triples_ids = read_cache(self.cache_path.cache_test_rel_triples_ids_path)
        self.valid_rel_triples_ids = read_cache(self.cache_path.cache_valid_rel_triples_ids_path)
        self.all_entities = read_cache(self.cache_path.cache_all_entities_path)
        self.all_attributes = read_cache(self.cache_path.cache_all_attributes_path)
        self.all_relations = read_cache(self.cache_path.cache_all_relations_path)
        self.all_values = read_cache(self.cache_path.cache_all_values_path)
        self.all_timestamps = read_cache(self.cache_path.cache_all_timestamps_path)
        self.entities_ids = read_cache(self.cache_path.cache_entities_ids_path)
        self.attributes_ids = read_cache(self.cache_path.cache_attributes_ids_path)
        self.relations_ids = read_cache(self.cache_path.cache_relations_ids_path)
        self.values_ids = read_cache(self.cache_path.cache_values_ids_path)
        self.timestamps_ids = read_cache(self.cache_path.cache_timestamps_ids_path)
        self.idx2entity = read_cache(self.cache_path.cache_idx2entity_path)
        self.idx2attribute = read_cache(self.cache_path.cache_idx2attribute_path)
        self.idx2relation = read_cache(self.cache_path.cache_idx2relation_path)
        self.idx2value = read_cache(self.cache_path.cache_idx2value_path)
        self.idx2timestamp = read_cache(self.cache_path.cache_idx2timestamp_path)
        self.entity2idx = read_cache(self.cache_path.cache_entity2idx_path)
        self.attribute2idx = read_cache(self.cache_path.cache_attribute2idx_path)
        self.relation2idx = read_cache(self.cache_path.cache_relation2idx_path)
        self.value2idx = read_cache(self.cache_path.cache_value2idx_path)
        self.timestamp2idx = read_cache(self.cache_path.cache_timestamp2idx_path)

        self.train_queries_answers = read_cache(self.cache_path.cache_train_queries_answers_path)
        self.valid_queries_answers = read_cache(self.cache_path.cache_valid_queries_answers_path)
        self.test_queries_answers = read_cache(self.cache_path.cache_test_queries_answers_path)

        meta = read_cache(self.cache_path.cache_metadata_path)
        self.read_meta(meta)

    def patch(self):
        self.restore_from_cache()
        self.sampling()
        self.cache_sampling_data()

    def load_cache_by_tasks(self, tasks: List[str], split="train"):
        qa = {}
        for query_name in tasks:
            path = self.cache_path.cache_queries_answers_path(split, query_name)
            if not path.exists():
                print(f"not cache exists for {query_name} in {split}")
                continue
            query_data = read_cache(path)
            qa[query_name] = query_data
        return qa

    def transform_all_data(self):
        NumericalKnowledgeData.transform_all_data(self)
        self.sampling()

    def sampling(self):
        # 0. prepare data.
        # add inverse relations
        max_attribute_id = self.attribute_count
        max_relation_id = self.relation_count
        attributes_ids_with_reverse = self.attributes_ids + [a + max_attribute_id for a in self.attributes_ids]
        relations_ids_with_reverse = self.relations_ids + [r + max_relation_id for r in self.relations_ids]

        # def append_reverse(att_triples, rel_triples):
        #     nonlocal max_attribute_id
        #     nonlocal max_relation_id
        #     att_res = []
        #     rel_res = []
        #     for s, a, x, t in att_triples:
        #         att_res.append((s, a, x, t))
        #         att_res.append((x, a + max_attribute_id, s, t))
        #     for s, r, o, t in rel_triples:
        #         rel_res.append((s, r, o, t))
        #         rel_res.append((o, r + max_relation_id, s, t))
        #     return (att_res, rel_res)

        # train_att_triples_ids, train_rel_triples_ids = append_reverse(self.train_att_triples_ids, self.train_rel_triples_ids)
        # valid_att_triples_ids, valid_rel_triples_ids = append_reverse(self.valid_att_triples_ids, self.valid_rel_triples_ids)
        # test_att_triples_ids, test_rel_triples_ids = append_reverse(self.test_att_triples_ids, self.test_rel_triples_ids)

        def append_reverse(rel_triples):
            nonlocal max_relation_id
            rel_res = []
            for s, r, o, t in rel_triples:
                rel_res.append((s, r, o, t))
                rel_res.append((o, r + max_relation_id, s, t))
            return (rel_res)

        train_rel_triples_ids = append_reverse(self.train_rel_triples_ids)
        valid_rel_triples_ids = append_reverse(self.valid_rel_triples_ids)
        test_rel_triples_ids = append_reverse(self.test_rel_triples_ids)

        train_att_triples_ids = self.train_att_triples_ids
        valid_att_triples_ids = self.valid_att_triples_ids
        test_att_triples_ids = self.test_att_triples_ids

        # 1. 1-hop: PAx, PAt, PAe
        train_sax_t, train_sat_x, train_xat_s = build_map_sax2t_and_sat2x_and_xat2s(self.train_att_triples_ids)
        valid_sax_t, valid_sat_x, valid_xat_s = build_map_sax2t_and_sat2x_and_xat2s(self.valid_att_triples_ids)
        test_sax_t, test_sat_x, test_xat_s = build_map_sax2t_and_sat2x_and_xat2s(self.test_att_triples_ids)

        # 1. 1-hop: PEe, PEt
        train_sro_t, train_srt_o = build_map_sax2t_and_sat2x(self.train_rel_triples_ids)
        valid_sro_t, valid_srt_o = build_map_sax2t_and_sat2x(self.valid_rel_triples_ids)
        test_sro_t, test_srt_o = build_map_sax2t_and_sat2x(self.test_rel_triples_ids)

        def build_one_hop(param_name_list: List[str], sax_t, for_test=False):
            queries_answers = []
            for s in sax_t:
                for a in sax_t[s]:
                    for x in sax_t[s][a]:
                        answers = sax_t[s][a][x]
                        if len(answers) > 0:
                            queries = [s, a, x]
                            if for_test:
                                queries_answers.append((queries, {}, answers))
                            else:
                                queries_answers.append((queries, answers))
            return {
                "args": param_name_list,
                "queries_answers": queries_answers
            }

        if self.cache_path.cache_train_queries_answers_path.exists():
            self.train_queries_answers = read_cache(self.cache_path.cache_train_queries_answers_path)
            self.valid_queries_answers = read_cache(self.cache_path.cache_valid_queries_answers_path)
            self.test_queries_answers = read_cache(self.cache_path.cache_test_queries_answers_path)

        def cache_step():
            cache_data(self.train_queries_answers, self.cache_path.cache_train_queries_answers_path)
            cache_data(self.valid_queries_answers, self.cache_path.cache_valid_queries_answers_path)
            cache_data(self.test_queries_answers, self.cache_path.cache_test_queries_answers_path)

        if "PAx" not in self.train_queries_answers:
            self.train_queries_answers["PAx"] = build_one_hop(["e1", "a1", "t1"], train_sat_x, for_test=False)
            cache_data(self.train_queries_answers["PAx"], self.cache_path.cache_queries_answers_path("train", "PAx"))
        if "PAx" not in self.valid_queries_answers:
            self.valid_queries_answers["PAx"] = build_one_hop(["e1", "a1", "t1"], valid_sat_x, for_test=True)
            cache_data(self.valid_queries_answers["PAx"], self.cache_path.cache_queries_answers_path("valid", "PAx"))
        if "PAx" not in self.test_queries_answers:
            self.test_queries_answers["PAx"] = build_one_hop(["e1", "a1", "t1"], test_sat_x, for_test=True)
            cache_data(self.test_queries_answers["PAx"], self.cache_path.cache_queries_answers_path("test", "PAx"))
        print("PAx",
              "train", len(self.train_queries_answers["PAx"]["queries_answers"]),
              "valid", len(self.valid_queries_answers["PAx"]["queries_answers"]),
              "test", len(self.test_queries_answers["PAx"]["queries_answers"]),
              )

        if "PAt" not in self.train_queries_answers:
            self.train_queries_answers["PAt"] = build_one_hop(["e1", "a1", "x1"], train_sax_t, for_test=False)
            cache_data(self.train_queries_answers["PAt"], self.cache_path.cache_queries_answers_path("train", "PAt"))
        if "PAt" not in self.valid_queries_answers:
            self.valid_queries_answers["PAt"] = build_one_hop(["e1", "a1", "x1"], valid_sax_t, for_test=True)
            cache_data(self.valid_queries_answers["PAt"], self.cache_path.cache_queries_answers_path("valid", "PAt"))
        if "PAt" not in self.test_queries_answers:
            self.test_queries_answers["PAt"] = build_one_hop(["e1", "a1", "x1"], test_sax_t, for_test=True)
            cache_data(self.test_queries_answers["PAt"], self.cache_path.cache_queries_answers_path("test", "PAt"))
        print("PAt",
              "train", len(self.train_queries_answers["PAt"]["queries_answers"]),
              "valid", len(self.valid_queries_answers["PAt"]["queries_answers"]),
              "test", len(self.test_queries_answers["PAt"]["queries_answers"]),
              )
        
        if "PAe" not in self.train_queries_answers:
            self.train_queries_answers["PAe"] = build_one_hop(["x1", "a1", "t1"], train_xat_s, for_test=False)
            cache_data(self.train_queries_answers["PAe"], self.cache_path.cache_queries_answers_path("train", "PAe"))
        if "PAe" not in self.valid_queries_answers:
            self.valid_queries_answers["PAe"] = build_one_hop(["x1", "a1", "t1"], valid_xat_s, for_test=True)
            cache_data(self.valid_queries_answers["PAe"], self.cache_path.cache_queries_answers_path("valid", "PAe"))
        if "PAe" not in self.test_queries_answers:
            self.test_queries_answers["PAe"] = build_one_hop(["x1", "a1", "t1"], test_xat_s, for_test=True)
            cache_data(self.test_queries_answers["PAe"], self.cache_path.cache_queries_answers_path("test", "PAe"))
        print("PAe",
              "train", len(self.train_queries_answers["PAe"]["queries_answers"]),
              "valid", len(self.valid_queries_answers["PAe"]["queries_answers"]),
              "test", len(self.test_queries_answers["PAe"]["queries_answers"]),
              )
        
        if "PRe" not in self.train_queries_answers:
            self.train_queries_answers["PRe"] = build_one_hop(["e1", "r1", "t1"], train_srt_o, for_test=False)
            cache_data(self.train_queries_answers["PRe"], self.cache_path.cache_queries_answers_path("train", "PRe"))
        if "PRe" not in self.valid_queries_answers:
            self.valid_queries_answers["PRe"] = build_one_hop(["e1", "r1", "t1"], valid_srt_o, for_test=True)
            cache_data(self.valid_queries_answers["PRe"], self.cache_path.cache_queries_answers_path("valid", "PRe"))
        if "PRe" not in self.test_queries_answers:
            self.test_queries_answers["PRe"] = build_one_hop(["e1", "r1", "t1"], test_srt_o, for_test=True)
            cache_data(self.test_queries_answers["PRe"], self.cache_path.cache_queries_answers_path("test", "PRe"))
        print("PRe",
              "train", len(self.train_queries_answers["PRe"]["queries_answers"]),
              "valid", len(self.valid_queries_answers["PRe"]["queries_answers"]),
              "test", len(self.test_queries_answers["PRe"]["queries_answers"]),
              )

        if "PRt" not in self.train_queries_answers:
            self.train_queries_answers["PRt"] = build_one_hop(["e1", "r1", "e2"], train_sro_t, for_test=False)
            cache_data(self.train_queries_answers["PRt"], self.cache_path.cache_queries_answers_path("train", "PRt"))
        if "PRt" not in self.valid_queries_answers:
            self.valid_queries_answers["PRt"] = build_one_hop(["e1", "r1", "e2"], valid_sro_t, for_test=True)
            cache_data(self.valid_queries_answers["PRt"], self.cache_path.cache_queries_answers_path("valid", "PRt"))
        if "PRt" not in self.test_queries_answers:
            self.test_queries_answers["PRt"] = build_one_hop(["e1", "r1", "e2"], test_sro_t, for_test=True)
            cache_data(self.test_queries_answers["PRt"], self.cache_path.cache_queries_answers_path("test", "PRt"))
        print("PRt",
              "train", len(self.train_queries_answers["PRt"]["queries_answers"]),
              "valid", len(self.valid_queries_answers["PRt"]["queries_answers"]),
              "test", len(self.test_queries_answers["PRt"]["queries_answers"]),
              )

        # 2. multi-hop: Pe_aPt, Pe_bPt, etc
        train_sax_t, train_sxa_t, train_sat_x, train_sta_x, \
        train_xas_t, train_xat_s, train_xta_s, \
        train_tas_x, train_tax_s, \
        train_ast_x, train_axt_s, train_asx_t, \
        train_t_sax, train_s_xat, train_x_sat = build_mapping_simple_att(train_att_triples_ids)
        
        valid_sax_t, valid_sxa_t, valid_sat_x, valid_sta_x, \
        valid_xas_t, valid_xat_s, valid_xta_s, \
        valid_tas_x, valid_tax_s, \
        valid_ast_x, valid_axt_s, valid_asx_t, \
        valid_t_sax, valid_s_xat, valid_x_sat = build_mapping_simple_att(train_att_triples_ids + valid_att_triples_ids)
        
        test_sax_t, test_sxa_t, test_sat_x, test_sta_x, \
        test_xas_t, test_xat_s, test_xta_s, \
        test_tas_x, test_tax_s, \
        test_ast_x, test_axt_s, test_asx_t, \
        test_t_sax, test_s_xat, test_x_sat = build_mapping_simple_att(train_att_triples_ids + valid_att_triples_ids + test_att_triples_ids)
        
        # 2.1 parser
        train_sro_t, train_sor_t, train_srt_o, train_str_o, \
        train_ors_t, train_trs_o, train_tro_s, train_rst_o, \
        train_rso_t, train_t_sro, train_o_srt = build_mapping_simple_rel(train_rel_triples_ids)
        valid_sro_t, valid_sor_t, valid_srt_o, valid_str_o, \
        valid_ors_t, valid_trs_o, valid_tro_s, valid_rst_o, \
        valid_rso_t, valid_t_sro, valid_o_srt = build_mapping_simple_rel(train_rel_triples_ids + valid_rel_triples_ids)
        test_sro_t, test_sor_t, test_srt_o, test_str_o, \
        test_ors_t, test_trs_o, test_tro_s, test_rst_o, \
        test_rso_t, test_t_sro, test_o_srt = build_mapping_simple_rel(train_rel_triples_ids + valid_rel_triples_ids + test_rel_triples_ids)

        # 2.1. parser
        train_parser = SamplingParser(self.entities_ids, relations_ids_with_reverse, attributes_ids_with_reverse, self.values_ids, self.timestamps_ids,
                                                train_sax_t, train_sxa_t, train_sat_x, train_sta_x, train_xas_t, train_xat_s, train_xta_s, train_tas_x, train_tax_s,
                                                train_ast_x, train_axt_s, train_asx_t, train_t_sax, train_s_xat, train_x_sat, train_sro_t, train_sor_t, train_srt_o, train_str_o, 
                                                train_ors_t, train_trs_o, train_tro_s, train_rst_o, train_rso_t, train_t_sro, train_o_srt)
        valid_parser = SamplingParser(self.entities_ids, relations_ids_with_reverse, attributes_ids_with_reverse, self.values_ids, self.timestamps_ids,
                                                valid_sax_t, valid_sxa_t, valid_sat_x, valid_sta_x, valid_xas_t, valid_xat_s, valid_xta_s, valid_tas_x, valid_tax_s,
                                                valid_ast_x, valid_axt_s, valid_asx_t, valid_t_sax, valid_s_xat, valid_x_sat, valid_sro_t, valid_sor_t, valid_srt_o, valid_str_o, 
                                                valid_ors_t, valid_trs_o, valid_tro_s, valid_rst_o, valid_rso_t, valid_t_sro, valid_o_srt)
        test_parser = SamplingParser(self.entities_ids, relations_ids_with_reverse, attributes_ids_with_reverse, self.values_ids, self.timestamps_ids,
                                                test_sax_t, test_sxa_t, test_sat_x, test_sta_x, test_xas_t, test_xat_s, test_xta_s, test_tas_x, test_tax_s,
                                                test_ast_x, test_axt_s, test_asx_t, test_t_sax, test_s_xat, test_x_sat, test_sro_t, test_sor_t, test_srt_o, test_str_o, 
                                                test_ors_t, test_trs_o, test_tro_s, test_rst_o, test_rso_t, test_t_sro, test_o_srt)

        # 2.2. sampling
        # we generate 1p, t-1p according to original train/valid/test triples.
        # for union-DM, we don't need to actually generate it.
        # The model should use 2u, up, t-2u, t-up with DM by itself.
        query_structure_name_list = [
                "PAe", "PRe", "PRe2", "ea2i", "er2i",
                "PAt", "PRt", "PRta", "PRtb", "ta2i", "tr2i",
                "PAx", "gPAx", "sPAx", "gPAxi", "sPAxi", "gsPAxi",
                "ea2u", "er2u",
                "ta2u", "tr2u",
                "gPAxu", "sPAxu", "gsPAxu",
        ] 
        
        # how many samples should we generate?
        max_sample_count = 5000 # len(build_map_srt_o(train_rel_triples_ids))
        train_sample_counts = {
            "PAe" : max_sample_count,
            "PRe" : max_sample_count,
            "PRe2" : max_sample_count,
            "ea2i" : max_sample_count,
            "er2i" : max_sample_count,

            "PAt" : max_sample_count,  
            "PRt" : max_sample_count,
            "PRta" : max_sample_count, 
            "PRtb" : max_sample_count, 
            "ta2i" : max_sample_count,
            "tr2i" : max_sample_count,

            "PAx" : max_sample_count, 
            "gPAx" : max_sample_count, 
            "sPAx" : max_sample_count, 
            "gPAxi" : max_sample_count, 
            "sPAxi" : max_sample_count, 
            "gsPAxi" : max_sample_count,

            #"ea2u" : max_sample_count,
            #"er2u" : max_sample_count,

            #"ta2u" : max_sample_count,
            #"tr2u" : max_sample_count, 

            #"gPAxu" : max_sample_count, 
            #"sPAxu" : max_sample_count, 
            #"gsPAxu" : max_sample_count

        }

        test_sample_count = 5000 # min(max_sample_count // 30, 10000)

        test_sample_counts = {
            "PAe" : test_sample_count,
            "PRe" : test_sample_count,
            "PRe2" : test_sample_count,
            "ea2i" : test_sample_count,
            "er2i" : test_sample_count,

            "PAt" : test_sample_count,  
            "PRt" : test_sample_count,
            "PRta" : test_sample_count, 
            "PRtb" : test_sample_count, 
            "ta2i" : test_sample_count,
            "tr2i" : test_sample_count,

            "PAx" : test_sample_count, 
            "gPAx" : test_sample_count, 
            "sPAx" : test_sample_count, 
            "gPAxi" : test_sample_count, 
            "sPAxi" : test_sample_count, 
            "gsPAxi" : test_sample_count,

            "ea2u" : test_sample_count,
            "er2u" : test_sample_count,

            "ta2u" : test_sample_count,
            "tr2u" : test_sample_count, 

            "gPAxu" : test_sample_count, 
            "sPAxu" : test_sample_count, 
            "gsPAxu" : test_sample_count
        }
        

        def achieve_answers(train_query_structure_func, valid_query_structure_func, test_query_structure_func, for_test=False):
            answers = set()
            valid_answers = set()
            test_answers = set()
            conflict_count = -1
            placeholders = get_my_placeholder_list(train_query_structure_func)
            while len(answers) <= 0 or (len(answers) > 0 and (len(valid_answers) <= 0 or len(test_answers) <= 0)):
                # len(answers) > 0 and (len(valid_answers) <= 0 or len(test_answers) <= 0)
                # for queries containing negation, test may has no answers while train has lots of answers.
                # if test has no answers, we are not able to calculate metrics.
                clear_my_placeholder_list(placeholders)
                sampling_query_answers: QuerySet = train_query_structure_func(*placeholders)
                if sampling_query_answers.ids is not None and len(sampling_query_answers.ids) > 0:
                    answers = sampling_query_answers.ids
                    fixed = my_placeholder2fixed(placeholders)
                    valid_answers = valid_query_structure_func(*fixed).ids
                    if for_test and len(valid_answers) <= len(answers) and conflict_count < 100:
                        answers = set()
                    test_answers = test_query_structure_func(*fixed).ids
                else:
                    answers = set()
                    valid_answers = set()
                    test_answers = set()
                conflict_count += 1
            # if conflict_count > 0:
            #     print("conflict_count=", conflict_count)
            queries = my_placeholder2sample(placeholders)
            return queries, answers, valid_answers, test_answers, conflict_count

        for query_structure_name in query_structure_name_list:
            print(query_structure_name)
            train_func = train_parser.eval(query_structure_name)
            param_name_list = get_param_name_list(train_func)
            train_queries_answers = []
            valid_queries_answers = []
            test_queries_answers = []

            fast_query_structure_name = f"fast_{query_structure_name}"
            if fast_query_structure_name in train_parser.fast_ops.keys():
                # fast sampling
                # the fast function is the proxy of the original function.
                # the fast function makes sure that len(answers)>0 with least steps (in one step if possible).
                sample_train_func = train_parser.eval(fast_query_structure_name)
            else:
                sample_train_func = train_parser.eval(query_structure_name)
            sample_valid_func = valid_parser.eval(query_structure_name)
            sample_test_func = test_parser.eval(query_structure_name)

            # 1. sampling train dataset
            if query_structure_name in train_sample_counts and query_structure_name not in self.train_queries_answers:
                sample_count = train_sample_counts[query_structure_name]
                bar = Progbar(sample_count)
                for i in range(sample_count):
                    queries, answers, valid_answers, test_answers, conflict_count = achieve_answers(
                        sample_train_func,
                        sample_valid_func,
                        sample_test_func,
                        for_test=False)
                    if None in queries:
                        raise Exception("In " + query_structure_name + ", queries contains None: " + str(queries))
                    train_queries_answers.append((queries, answers))
                    if len(valid_answers) > len(answers):
                        valid_queries_answers.append((queries, answers, valid_answers))
                    if len(test_answers) > len(answers):
                        test_queries_answers.append((queries, answers, test_answers))
                    bar.update(i + 1, {"train": len(answers), "valid": len(valid_answers), "test": len(test_answers)})
                self.train_queries_answers[query_structure_name] = {
                    "args": param_name_list,
                    "queries_answers": train_queries_answers
                }
                cache_data(self.train_queries_answers[query_structure_name], self.cache_path.cache_queries_answers_path("train", query_structure_name))

            # 2. sampling valid/test dataset
            if query_structure_name in test_sample_counts and query_structure_name not in self.valid_queries_answers:
                sample_count = test_sample_counts[query_structure_name]
                bar = Progbar(sample_count)
                conflict_patient = 0
                for i in range(sample_count):
                    queries, answers, valid_answers, test_answers, conflict_count = achieve_answers(
                        sample_train_func,
                        sample_valid_func,
                        sample_test_func,
                        for_test=conflict_patient <= 100)
                    if conflict_patient <= 100 and conflict_count >= 99 and i <= 1000:
                        conflict_patient += 1
                    valid_queries_answers.append((queries, answers, valid_answers))
                    test_queries_answers.append((queries, answers, test_answers))
                    bar.update(i + 1, {"train": len(answers), "valid": len(valid_answers), "test": len(test_answers)})
                    if len(valid_queries_answers) >= 10000 and len(test_queries_answers) >= 10000:
                        valid_queries_answers = valid_queries_answers[:10000]
                        test_queries_answers = test_queries_answers[:10000]
                        bar.update(sample_count, {"train": len(answers), "valid": len(valid_answers), "test": len(test_answers)})
                        break
                self.valid_queries_answers[query_structure_name] = {
                    "args": param_name_list,
                    "queries_answers": valid_queries_answers
                }
                self.test_queries_answers[query_structure_name] = {
                    "args": param_name_list,
                    "queries_answers": test_queries_answers
                }
                cache_data(self.valid_queries_answers[query_structure_name], self.cache_path.cache_queries_answers_path("valid", query_structure_name))
                cache_data(self.test_queries_answers[query_structure_name], self.cache_path.cache_queries_answers_path("test", query_structure_name))

        # 3. calculate meta
        def avg_answers_count(qa):
            return sum([len(row[-1]) for row in qa]) / len(qa) if len(qa) > 0 else 0

        for query_name in self.test_queries_answers.keys():
            if query_name == "PRe" or query_name == "PRt" or query_name == "PAe" or query_name == "PAt" or query_name == "PAx":
                continue
            valid_qa = self.valid_queries_answers[query_name]["queries_answers"] if query_name in self.valid_queries_answers else []
            test_qa = self.test_queries_answers[query_name]["queries_answers"] if query_name in self.test_queries_answers else []
            self.valid_queries_answers[query_name]["queries_answers"] = valid_qa[:10000]
            self.test_queries_answers[query_name]["queries_answers"] = test_qa[:10000]
            cache_data(self.valid_queries_answers[query_name], self.cache_path.cache_queries_answers_path("valid", query_name))
            cache_data(self.test_queries_answers[query_name], self.cache_path.cache_queries_answers_path("test", query_name))

        for query_name in self.test_queries_answers.keys():
            train_qa = self.train_queries_answers[query_name]["queries_answers"] if query_name in self.train_queries_answers else []
            valid_qa = self.valid_queries_answers[query_name]["queries_answers"] if query_name in self.valid_queries_answers else []
            test_qa = self.test_queries_answers[query_name]["queries_answers"] if query_name in self.test_queries_answers else []
            queries_answers = train_qa + valid_qa + test_qa
            self.query_meta[query_name] = {
                "queries_count": len(queries_answers),
                "avg_answers_count": avg_answers_count(queries_answers),
                "train": {
                    "queries_count": len(train_qa),
                    "avg_answers_count": avg_answers_count(train_qa),
                },
                "valid": {
                    "queries_count": len(valid_qa),
                    "avg_answers_count": avg_answers_count(valid_qa),
                },
                "test": {
                    "queries_count": len(test_qa),
                    "avg_answers_count": avg_answers_count(test_qa),
                },
            }
            print(query_name, self.query_meta[query_name])

    def cache_all_data(self):
        NumericalKnowledgeData.cache_all_data(self)
        self.cache_sampling_data()

    def cache_sampling_data(self):
        cache_data(self.train_queries_answers, self.cache_path.cache_train_queries_answers_path)
        cache_data(self.valid_queries_answers, self.cache_path.cache_valid_queries_answers_path)
        cache_data(self.test_queries_answers, self.cache_path.cache_test_queries_answers_path)
        cache_data(self.meta(), self.cache_path.cache_metadata_path)

    def read_meta(self, meta):
        NumericalKnowledgeData.read_meta(self, meta)
        self.query_meta = meta["query_meta"]

    def meta(self) -> Dict[str, Any]:
        m = NumericalKnowledgeData.meta(self)
        m.update({
            "query_meta": self.query_meta,
        })
        return m

    def dump(self) -> List[str]:
        """ Function to dump statistic information of a dataset """
        # dump key information
        dump = NumericalKnowledgeData.dump(self)
        for k, v in self.query_meta.items():
            dump.insert(len(dump) - 2, f"{k} : {pformat(v)}")
        return dump


groups = {
    "avg_e": ["PAe", "PRe", "PRe2", "ea2i", "er2i"],
    "avg_t": ["PAt", "PRt", "PRta", "PRtb", "ta2i", "tr2i"],
    "avg_x": ["PAx", "gPAx", "sPAx", "gPAxi", "sPAxi", "gsPAxi"],
    "avg_Ue": ["ea2u", "er2u"],
    "avg_Ut": ["tr2u","ta2u"],
    "avg_Ux": ["gPAxu","sPAxu","gsPAxu"],
    "relations": ["P54", 
        "P1087", 
        "P1082", 
         "P131", 
         "P1411", 
         "P166",
         "P2962",
         "P17",
         "P108",
         "P4080",
         "P1352",
         "P26", 
         "P1343", 
         "P31", 
         "P2656",
         "P39", 
         "P1831",
         "P2645", 
         "P551", 
         "P2299",
         "P793",
         "P463",
         "P1538", 
         "P1346", 
         "P2196", 
         "P69", 
         "P1198", 
         "P176", 
         "P945",
         "P2046", 
         "P138", 
         "P512", 
         "P36", 
         "P1540", 
         "P190", 
         "P35", 
         "P137", 
         "P1923", 
         "P1454", 
         "P1098",
         "P710",
         "P27", 
         "P6", 
         "P1891", 
         "P102",
         "P1435", 
         "P1308",
         "P156",
         "P150",
         "P106",
         "P159",
         "P1539",
         "P1436",
         "P1603",
         "P488",
         "P1120",
         "P527",
         "P937", 
         "P1376", 
         "P1001", 
         "P286", 
         "P1971", 
         "P1336", 
         "P3975", 
         "P119", 
         "P97", 
         "P2997", 
         "P47", 
         "P608", 
         "P2067", 
         "P1344",
         "P1128", 
         "P609", 
         "P38", 
         "P127", 
         "P3000", 
         "P2437",
         "P1998", 
         "P276", 
         "P410", 
         "P647",
         "P449",
         "P361",
         "P411",
         "P3643",
         "P1351",
         "P195",
         "P1113",
         "P530",
         "P2415",
         "P439",
         "P4202",
         "P2388",
         "P2522", 
         "P414", 
         "P466"]
}