import logging
from backend.app.graph.connection import neo4j_conn

logger = logging.getLogger("graphrag.graph.schema")

CONSTRAINTS = [
    "CREATE CONSTRAINT drug_id_unique IF NOT EXISTS FOR (d:Drug) REQUIRE d.drug_id IS UNIQUE;",
    "CREATE CONSTRAINT condition_id_unique IF NOT EXISTS FOR (c:Condition) REQUIRE c.condition_id IS UNIQUE;",
    "CREATE CONSTRAINT symptom_id_unique IF NOT EXISTS FOR (s:Symptom) REQUIRE s.symptom_id IS UNIQUE;",
    "CREATE CONSTRAINT sideeffect_id_unique IF NOT EXISTS FOR (se:SideEffect) REQUIRE se.side_effect_id IS UNIQUE;",
    "CREATE CONSTRAINT population_id_unique IF NOT EXISTS FOR (p:Population) REQUIRE p.population_id IS UNIQUE;",
    "CREATE CONSTRAINT study_id_unique IF NOT EXISTS FOR (st:ClinicalStudy) REQUIRE st.study_id IS UNIQUE;",
    "CREATE CONSTRAINT guideline_id_unique IF NOT EXISTS FOR (g:Guideline) REQUIRE g.guideline_id IS UNIQUE;"
]

def initialize_constraints() -> bool:
    """Executes schema constraints creation queries in Neo4j."""
    try:
        driver = neo4j_conn.get_driver()
        with driver.session() as session:
            for constraint_cypher in CONSTRAINTS:
                logger.info(f"Running constraint initialization: {constraint_cypher.strip()}")
                session.run(constraint_cypher)
        logger.info("Successfully initialized all Neo4j schema constraints.")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Neo4j schema constraints: {str(e)}")
        logger.warning("Application will proceed without graph constraints. Ensure Neo4j is running.")
        return False
