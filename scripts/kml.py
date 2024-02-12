import pandas as pd
from xml.etree.ElementTree import Element, SubElement, tostring

colors = ["#ffffe5ff", "#f7fcb9ff", "#d9f0a3ff", "#addd8eff", "#78c679ff", "#41ab5dff", "#238443ff", "#006837ff", "#09522cff", "#023827ff"]
bin_count = len(colors)
housing = pd.read_csv("./datasets/housing.csv")
kml_set = housing[["id", "lat", "long", "price"]]
kml_set["color"] = pd.cut(kml_set["price"], bins=bin_count, labels=colors)
kml_set["type"] = "circle"

kml_set.to_csv("./datasets/priceloc.csv", columns=["lat", "long", "color", "type", "id"], index=False)

# xml = Element('kml')
# document = SubElement(xml, 'Document')
# name = SubElement(document, 'name')
# name.text = "Housing Prices"
# description = SubElement(document, 'description')
# description.text = "Prices for houses in module 3 dataset"
# for _, row in kml_set.head().iterrows():
#     print(row["lat"])

# file = open('./datasets/housing.xml', 'w')
# file.write(str(tostring(xml)))
# file.close()