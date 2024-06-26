# TODO add to reqs
from more_itertools import one
import pydantic_numpy


class Xsd:
    # TODO use jinja?
    header = """<?xml version="1.0" encoding="UTF-8" ?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
"""
    footer = """</xs:schema>"""

    def __init__(self):
        self.custom_types = []
        self.custom_type_names = []
        self.elements = []

    def __str__(self):
        return "\n".join([
            Xsd.header,
            *self.custom_types,
            '<xs:element name="ssrdata"><xs:complexType><xs:sequence>',
            *self.elements,
            '</xs:sequence></xs:complexType></xs:element>',
            Xsd.footer,
        ])

    @staticmethod
    def clean(xml):
        return "\n".join(s for s in xml.split("\n") if s.strip()) + "\n"

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


def create_type(xsd, base_type, ge=None, le=None):
    if ge is not None:
        if ge == 0 and base_type == int:
            return "positiveInteger"
        elif base_type in [int, float]:
            return xsd.add_type(base_type=base_type, ge=ge, le=le)
        else:
            raise ValueError(f"unsupported base_type {base_type}. please request")

    else:
        raise ValueError("not implemented. please request.")


def get_xsd_type(field, xsd):
    if field.annotation == int:
        xsd_type = "integer"
        for metadatum in field.metadata:
            if type(metadatum).__name__ == "Ge":
                xsd_type = create_type(xsd=xsd, base_type=field.annotation, ge=metadatum.ge)
            else:
                raise valueerror(f"unsupported field metadata: {metadatum}. "
                        "please request."
                        )
    elif field.annotation == str:
        xsd_type = "string"
    elif field.annotation == float:
        xsd_type = "decimal"
        for metadatum in field.metadata:
            if type(metadatum).__name__ == "Ge":
                xsd_type = create_type(xsd=xsd, base_type=field.annotation, ge=metadatum.ge)
            else:
                raise ValueError(f"Unsupported field metadata: {metadatum}. "
                        "Please request."
                        )
    return ("" if xsd_type in xsd.custom_type_names else "xs:") + xsd_type

def create_element(name, field, xsd):
    if field.annotation in (int, float, str):
        element = f"""<xs:element name="{name}" type="{get_xsd_type(field, xsd=xsd)}"/>"""
    elif field.annotation == (pydantic_numpy.typing.NpNDArray).__origin__:
        element = f"""<xs:element name="{name}" type="xs:string"/>"""
    else:
        raise ValueError(f"Unsupported field type: {field.annotation}."
                "Please request"
                )
    return element


def model2xsd(model):
    """Convert a Pydantic `BaseModel` into an XML schema.


    Supported types: int, float, str, NumPy array.
    """
    xsd = Xsd()
    for name, field in model.model_fields.items():
        xsd.add_element(create_element(name=name, field=field, xsd=xsd))
    return str(xsd)


def model2xml(model):
    pass

