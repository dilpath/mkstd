import xmlschema
import xml.dom.minidom
from pydantic import BaseModel
import pydantic_numpy

from .standard import Standard


class XmlStandard(Standard):
    # TODO use jinja?
    header = """<?xml version="1.0" encoding="UTF-8" ?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
"""
    footer = """</xs:schema>"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_types = []
        self.custom_type_names = []
        self.elements = []
        
        self.parse_model()

    def parse_model(self):
        for name, field in self.model.model_fields.items():
            self.add_element(self.make_element(name=name, field=field))

    def create_type(self, base_type, ge=None, le=None):
        if ge is not None:
            if ge == 0 and base_type == int:
                return "positiveInteger"
            elif base_type in [int, float]:
                return self.add_type(base_type=base_type, ge=ge, le=le)
            else:
                raise ValueError(f"unsupported base_type {base_type}. please request")
    
        else:
            raise ValueError("not implemented. please request.")
    
    
    def get_type(self, field):
        if field.annotation == int:
            xsd_type = "integer"
            for metadatum in field.metadata:
                if type(metadatum).__name__ == "Ge":
                    xsd_type = self.create_type(base_type=field.annotation, ge=metadatum.ge)
                else:
                    raise ValueError(f"Unsupported field metadata: {metadatum}. Please request.")
        elif field.annotation == str:
            xsd_type = "string"
        elif field.annotation == float:
            xsd_type = "decimal"
            for metadatum in field.metadata:
                if type(metadatum).__name__ == "Ge":
                    xsd_type = self.create_type(base_type=field.annotation, ge=metadatum.ge)
                else:
                    raise ValueError(f"Unsupported field metadata: {metadatum}. Please request.")
        return ("" if xsd_type in self.custom_type_names else "xs:") + xsd_type

    def make_element(self, name, field):
        if field.annotation in (int, float, str):
            element = f"""<xs:element name="{name}" type="{self.get_type(field)}"/>"""
        elif field.annotation == (pydantic_numpy.typing.NpNDArray).__origin__:
            element = f"""<xs:element name="{name}" type="xs:string"/>"""
        else:
            raise ValueError(f"Unsupported field type: {field.annotation}. Please request")
        return element

    def add_element(self, element):
        self.elements.append(element)

    def add_type(self, base_type, ge=None, le=None):
        base_type_xsd = "integer" if base_type == int else "decimal"
        custom_type_name = base_type_xsd + "Ge" + str(ge)
        if custom_type_name in self.custom_type_names:
            return custom_type_name
        self.custom_types.append(f"""<xs:simpleType name="{custom_type_name}">
  <xs:restriction base="xs:{base_type_xsd}">
    <xs:minInclusive value="{ge}"/>
  </xs:restriction>
</xs:simpleType>""")
        self.custom_type_names.append(custom_type_name)
        return custom_type_name

    def get_schema(self):
        return "\n".join(
            line for line in xml.dom.minidom.parseString("".join([
                XmlStandard.header,
                *self.custom_types,
                '<xs:element name="ssrdata"><xs:complexType><xs:sequence>',
                *self.elements,
                '</xs:sequence></xs:complexType></xs:element>',
                XmlStandard.footer,
            ])).toprettyxml().split("\n")
            if line.strip()
        ) + "\n"

    def parse_data(self, data: BaseModel):
        xs = xmlschema.XMLSchema(self.get_schema())
        etree = xs.encode(data.model_dump())
        return xmlschema.etree_tostring(etree)

    def load_data(self, filename: str):
        with open(filename, 'r') as f:
            data = xmlschema.XMLSchema(self.get_schema()).decode(f.read())
        return self.model.parse_obj(data)

