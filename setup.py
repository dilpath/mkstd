import os
import re

from setuptools import setup


def read(fname):
    """Read a file."""
    return open(fname).read()


def absolute_links(txt):
    """Replace relative github links by absolute links."""
    raw_base = (
        "(https://raw.githubusercontent.com/dilpath/mkstd/default/"
    )
    embedded_base = (
        "(https://github.com/dilpath/mkstd/tree/default/"
    )
    # iterate over links
    for var in re.findall(r"\[.*?\]\((?!http).*?\)", txt):
        if re.match(r".*?.(png|svg)\)", var):
            # link to raw file
            rep = var.replace("(", raw_base)
        else:
            # link to github embedded file
            rep = var.replace("(", embedded_base)
        txt = txt.replace(var, rep)
    return txt


# project metadata
# noinspection PyUnresolvedReferences
setup(
    long_description=absolute_links(read("README.md")),
    long_description_content_type="text/markdown",
)

