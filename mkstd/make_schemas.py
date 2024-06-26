import json
import numpy as np
import xml.dom.minidom
from pathlib import Path

import xmlschema

from .data_model import SSRData
from .xsd import Xsd, model2xsd

expected_data = {
    "ssr_level": 1,
    "ssr_version": 3,
    #"variable_names": ["v1", "v2", "v3"],

    "simulation_times": np.linspace(0, 10, 11),
    "sample_size": 100000,
    "ecf_evals": np.eye(5),
    "ecf_tval": np.eye(3),
    "ecf_nval": 1000,

    "error_metric_mean": 0.5,
    "error_metric_stdev": 0.2,

    "sig_figs": 5,
}

test_xml_data = """<?xml version="1.0" ?>
<ssrdata>
<ssr_level>5</ssr_level>
<ssr_version>3</ssr_version>
<simulation_times>53fverv3</simulation_times>
  <sample_size>50</sample_size>
<ecf_evals>f45rfc</ecf_evals>
<ecf_tval>frtvr</ecf_tval>
  <ecf_nval>20</ecf_nval>
<error_metric_mean>43.0</error_metric_mean>
<error_metric_stdev>2.0</error_metric_stdev>
  <sig_figs>4</sig_figs>
</ssrdata>
"""


output_path = Path(__file__).parent.resolve() / "output"
output_path.mkdir(exist_ok=True, parents=True)

expected_model = SSRData(**expected_data)

expected_model.to_hdf5(output_path / "test_model.hdf5")
test_hdf5 = SSRData.from_hdf5(output_path / "test_model.hdf5")

expected_model.to_json(output_path / "test_model.json")
test_json = SSRData.from_json(output_path / "test_model.json")

expected_model.to_yaml(output_path / "test_model.yaml")
test_yaml = SSRData.from_yaml(output_path / "test_model.yaml")

#expected_model.to_xml(output_path / "test_model.xml")
#test_xml = SSRData.from_xml(output_path / "test_model.xml")

xs = model2xsd(SSRData)
xml_schema = xml.dom.minidom.parseString(xs).toprettyxml()
with open(output_path / "schema.xsd", 'w') as f:
    f.write(Xsd.clean(xml_schema))
#import subprocess
#subprocess.run(["generateDS.py", "-o", output_path/ "xml_class.py", output_path / "schema.xsd"])

schema = SSRData.model_json_schema()
with open(output_path / "schema.json", 'w') as f:
    f.write(json.dumps(schema, indent=2))


xsch = xmlschema.XMLSchema("output/schema.xsd")
parsed_xml = xsch.to_dict(test_xml_data)
SSRData.parse_obj(parsed_xml)
