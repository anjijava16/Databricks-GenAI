from dataclasses import dataclass

MAX_UC_ENTITY_NAME_LEN = 63  # max length of UC entity name


@dataclass(frozen=True)
class UnityCatalogEntity:
    """Helper data class representing a Unity Catalog entity.

    Attributes:
        catalog (str): The catalog name.
        schema (str): The schema name.
        entity (str): The entity name.
    """

    catalog: str
    schema: str
    entity: str

    @staticmethod
    def from_fullname(fullname: str):
        parts = fullname.split(".")
        # Remove any backticks wrapped around the UC securables.
        parts = [part[1:-1] if part.startswith("`") and part.endswith("`") else part for part in parts]
        assert len(parts) == 3

        return UnityCatalogEntity(catalog=parts[0], schema=parts[1], entity=parts[2])

    @property
    def fullname(self):
        return f"{self.catalog}.{self.schema}.{self.entity}"

    @property
    def fullname_with_backticks(self):
        catalog_name = self._sanitize_identifier(self.catalog)
        schema_name = self._sanitize_identifier(self.schema)
        entity_name = self._sanitize_identifier(self.entity)
        return f"{catalog_name}.{schema_name}.{entity_name}"

    @staticmethod
    def _sanitize_identifier(identifier: str) -> str:
        """
        Escape special characters and delimit an identifier with backticks.
        For example, "a`b" becomes "`a``b`".
        Use this function to sanitize identifiers such as table/column names in SQL and PySpark.
        """
        return f"`{identifier.replace('`', '``')}`"
