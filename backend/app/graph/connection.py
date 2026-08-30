import logging
from typing import Optional
# pyrefly: ignore [missing-import]
from neo4j import GraphDatabase, Driver
from backend.app.config import settings

logger = logging.getLogger("graphrag.graph.connection")

class Neo4jConnection:
    def __init__(self) -> None:
        self._driver: Optional[Driver] = None

    def get_driver(self) -> Driver:
        """Returns the Neo4j driver, initializing it if it does not exist."""
        if self._driver is None:
            try:
                self._driver = GraphDatabase.driver(
                    settings.NEO4J_URI,
                    auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
                )
                # Verify connection immediately
                self._driver.verify_connectivity()
                logger.info("Successfully connected to Neo4j database.")
            except Exception as e:
                logger.error(f"Failed to connect to Neo4j: {str(e)}")
                self._driver = None
                raise e
        return self._driver

    def close(self) -> None:
        """Closes the Neo4j driver instance."""
        if self._driver is not None:
            self._driver.close()
            logger.info("Neo4j database connection closed.")
            self._driver = None

    def verify_health(self) -> bool:
        """Checks if the Neo4j connection is active and healthy."""
        try:
            driver = self.get_driver()
            driver.verify_connectivity()
            return True
        except Exception:
            return False

# Global connection instance
neo4j_conn = Neo4jConnection()
