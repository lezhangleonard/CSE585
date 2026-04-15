from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
from copy import deepcopy
from llm_reasoning_engine import Decision
from neo4j import GraphDatabase

class GraphStore(ABC):
    """
    Abstract storage interface.
    Enables swapping backends (in-memory, Neo4j, etc.)
    """

    @abstractmethod
    def snapshot(self) -> Dict[Tuple[str, str], Optional[str]]:
        pass

    @abstractmethod
    def apply(self, decision: Decision) -> None:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    @abstractmethod
    def get_final_state(self) -> Dict[Tuple[str, str], Optional[str]]:
        pass


class InMemoryStore(GraphStore):
    """
    In-memory state representation:

        (subject, predicate) -> active_object

    If no active state exists, key is absent.
    """

    def __init__(self):
        self._state: Dict[Tuple[str, str], Optional[str]] = {}

    def snapshot(self) -> Dict[Tuple[str, str], Optional[str]]:
        """
        Returns deep copy to simulate isolation.
        Critical for batch executor correctness.
        """
        return deepcopy(self._state)

    def apply(self, decision: Decision) -> None:
        """
        Applies decision returned by reasoning engine.
        """
        if decision.action != "APPLY":
            return

        key = (decision.subject, decision.predicate)

        if decision.new_value is None:
            # Deactivation
            self._state.pop(key, None)
        else:
            # Activation / Switch
            self._state[key] = decision.new_value

    def clear(self) -> None:
        self._state.clear()

    def get_final_state(self) -> Dict[Tuple[str, str], Optional[str]]:
        """
        Returns deterministic sorted copy of state.
        Used for logging and correctness comparison.
        """
        return dict(sorted(self._state.items()))
    

class Neo4jStore(GraphStore):

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def clear(self) -> None:
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def snapshot(self) -> Dict[Tuple[str, str], Optional[str]]:
        """
        Fetch current active state from DB and return as dict.
        This is expensive but correct for baseline experiments.
        """
        query = """
        MATCH (s:Entity)-[r:REL]->(o:Entity)
        WHERE r.predicate IS NOT NULL
        RETURN s.name AS subject, r.predicate AS predicate, o.name AS object
        """

        result = {}

        with self.driver.session() as session:
            records = session.run(query)
            for record in records:
                key = (record["subject"], record["predicate"])
                result[key] = record["object"]

        return result

    def apply(self, decision: Decision) -> None:
        if decision.action != "APPLY":
            return

        s = decision.subject
        p = decision.predicate
        new_value = decision.new_value

        with self.driver.session() as session:

            # Remove existing relation for (s, p)
            delete_query = """
            MATCH (s:Entity {name: $subject})-[r:REL {predicate: $predicate}]->()
            DELETE r
            """
            session.run(delete_query, subject=s, predicate=p)

            # If new_value is not None, create new relation
            if new_value is not None:
                create_query = """
                MERGE (s:Entity {name: $subject})
                MERGE (o:Entity {name: $object})
                CREATE (s)-[:REL {predicate: $predicate}]->(o)
                """
                session.run(
                    create_query,
                    subject=s,
                    object=new_value,
                    predicate=p
                )

    def get_final_state(self) -> Dict[Tuple[str, str], Optional[str]]:
        return dict(sorted(self.snapshot().items()))