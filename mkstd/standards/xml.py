from typing import Any

import xmlschema
from defusedxml.minidom import parseString as parse_xml
from pydantic import BaseModel, Field

from ..types.array import array_to_string, is_array_type, string_to_array
from .standard import Standard


class XmlStandard(Standard):
    """The XML standard.

    Use this to create a standard in the JSON file format.

    See :class:`Standard` for inherited methods and attributes.

    Schema are generated in the XML schema (XSD) format.

    Attributes:
        custom_types:
            Definitions of custom types in the schema format.
        custom_type_names:
            The names of the custom types.
        elements:
            XML elements of the fields in the data model.
    """

    # TODO use jinja?
    header = """<?xml version="1.0" encoding="UTF-8" ?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
"""
    footer = """</xs:schema>"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.custom_types = []
        self.custom_type_names = []
        self.elements = self._generate_elements()

    def _generate_elements(self) -> list[str]:
        """Generate XML elements for the fields of the data model.

        Returns:
            The XML elements.
        """
        elements = []
        for name, field in self.model.model_fields.items():
            element = self._get_element(name=name, field=field)
            elements.append(element)
        return elements

    def _get_numeric_type(
        self,
        base_type: type[int] | type[float],
        ge: int | float = None,
        le: int | float = None,
    ) -> str:
        """Get the XML type name of a (constrained) numeric value.

        The type is created if no standard types are suitable.

        Args:
            base_type:
                The base Python type of the number.
            ge:
                The lower bound (inclusive); "greater than or equal to".
            le:
                The upper bound (inclusive); "less than or equal to".

        Returns:
            The XML type name.
        """
        if ge is not None:
            if ge == 0 and base_type == int:
                return "positiveInteger"
            elif base_type in [int, float]:
                return self._generate_numeric_type(
                    base_type=base_type, ge=ge, le=le
                )
            else:
                raise NotImplementedError(f"Base type `{base_type}`.")
        else:
            raise NotImplementedError("`ge is None`.")

    def _get_type(self, field: Field) -> str:
        """Get the XML type name of a data model field.

        Args:
            field:
                The field.

        Returns:
            The XML type name.
        """

        if field.annotation == int:
            xsd_type = "integer"
            for metadatum in field.metadata:
                if type(metadatum).__name__ == "Ge":
                    xsd_type = self._get_numeric_type(
                        base_type=field.annotation, ge=metadatum.ge
                    )
                else:
                    raise NotImplementedError(
                        f"Field metadata contains {metadatum}."
                    )
        elif field.annotation in (str, list[str]) or is_array_type(
            field.annotation
        ):
            xsd_type = "string"
        elif field.annotation == float:
            xsd_type = "decimal"
            for metadatum in field.metadata:
                if type(metadatum).__name__ == "Ge":
                    xsd_type = self._get_numeric_type(
                        base_type=field.annotation, ge=metadatum.ge
                    )
                else:
                    raise NotImplementedError(
                        f"Field metadata contains {metadatum}."
                    )
        else:
            raise NotImplementedError(f"Field type `{field.annotation}`.")
        return ("" if xsd_type in self.custom_type_names else "xs:") + xsd_type

    def _get_element(self, name: str, field: Field) -> str:
        """Get an XML element for a field.

        Args:
            name:
                The name of the XML element.
            field:
                The field.

        Returns:
            The XML element.
        """
        attr_name = f' name="{name}"'
        attr_type = f' type="{self._get_type(field)}"'
        attr_requirement = ""
        if not field.is_required():
            attr_requirement = ' minOccurs="0"'
        if field.annotation in (int, float, str, list[str]) or is_array_type(
            field.annotation
        ):
            element = f"<xs:element{attr_name}{attr_type}{attr_requirement}/>"
        else:
            raise NotImplementedError(f"Field type `{field.annotation}`.")
        return element

    def _generate_numeric_type(
        self,
        base_type: type[int] | type[float],
        ge: int | float = None,
        le: int | float = None,
    ) -> str:
        """Generate an XML type of a (constrained) numeric type.

        Args:
            base_type:
                The base Python type of the number.
            ge:
                The lower bound (inclusive); "greater than or equal to".
            le:
                The upper bound (inclusive); "less than or equal to".

        Returns:
            The name of the generated XML type.
        """
        base_type_xsd = "integer" if base_type == int else "decimal"
        custom_type_name = base_type_xsd + "Ge" + str(ge)
        if custom_type_name in self.custom_type_names:
            return custom_type_name
        self.custom_types.append(
            f"""<xs:simpleType name="{custom_type_name}">
  <xs:restriction base="xs:{base_type_xsd}">
    <xs:minInclusive value="{ge}"/>
  </xs:restriction>
</xs:simpleType>"""
        )
        self.custom_type_names.append(custom_type_name)
        return custom_type_name

    def get_schema(self) -> str:
        """See :class:`Standard`."""
        return (
            "\n".join(
                line
                for line in parse_xml(
                    "".join(
                        [
                            XmlStandard.header,
                            *self.custom_types,
                            '<xs:element name="ssrdata"><xs:complexType><xs:sequence>',
                            *self.elements,
                            "</xs:sequence></xs:complexType></xs:element>",
                            XmlStandard.footer,
                        ]
                    )
                )
                .toprettyxml()
                .split("\n")
                if line.strip()
            )
            + "\n"
        )

    def format_data(self, data: BaseModel) -> str:
        """See :class:`Standard`."""
        xs = xmlschema.XMLSchema(self.get_schema())
        dump = data.model_dump()
        _apply_converters(dump=dump, model=self.model)
        etree = xs.encode(dump)
        return xmlschema.etree_tostring(etree)

    def _load_data(self, filename: str) -> dict[str, Any]:
        """See :class:`Standard`."""
        with open(filename) as f:
            data = xmlschema.XMLSchema(self.get_schema()).decode(f.read())
        _apply_deconverters(dump=data, model=self.model)
        return data


def _apply_converters(dump: dict[str, Any], model: type[BaseModel]) -> None:
    """Convert Python types for serialization.

    Args:
        dump:
            The data, e.g. the output from :func:`BaseModel.model_dump`.
        model:
            The data model.
    """
    _convert_iterables(dump=dump, model=model)


def _apply_deconverters(dump: dict[str, Any], model: type[BaseModel]) -> None:
    """Convert serialized types for deserialization.

    Args:
        dump:
            The data, e.g. the output from :func:`BaseModel.model_dump`.
        model:
            The data model.
    """
    _deconvert_iterables(dump=dump, model=model)


def _convert_iterables(dump: dict[str, Any], model: type[BaseModel]) -> None:
    for field_name, field in model.model_fields.items():
        if field.annotation == list[str] or is_array_type(field.annotation):
            dump[field_name] = array_to_string(
                field_name=field_name, array=dump[field_name], model=model
            )


def _deconvert_iterables(dump: dict[str, Any], model: type[BaseModel]) -> None:
    for field_name, field in model.model_fields.items():
        if field.annotation == list[str] or is_array_type(field.annotation):
            dump[field_name] = string_to_array(
                field_name=field_name, array=dump[field_name], model=model
            )
