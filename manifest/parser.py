import json
import re

from lxml import etree

_parser = etree.XMLParser(
    resolve_entities=False,
    load_dtd=False,
    recover=True,
    encoding="utf-8",
)


def escape_entities(xml_str: str):
    """Replace unescaped ampersands with &amp; in an XML string. Lots of
    responses have amperands that are not escaped, which causes the XML parser
    to break."""
    # Pattern to find & that are not already part of an entity. Negative
    # lookbehind & lookahead assertions are used.
    pattern = re.compile(r"&(?!amp;|lt;|gt;|quot;|#)")
    # Replace those ampersands with &amp;
    return pattern.sub(r"&amp;", xml_str)


def parse_document(s: str) -> etree._Element:
    escaped = escape_entities(s)
    root = etree.fromstring(escaped, _parser)
    return root


def find_and_parse_xml(s: str, root_tag="root") -> etree._Element | None:
    """For a given string, find the first XML document and parse it."""
    start_tag = f"<{root_tag}>"
    end_tag = f"</{root_tag}>"

    start_index = s.find(start_tag)
    end_index = s.find(end_tag)

    # Couldn't find the root tag
    if start_index == -1 or end_index == -1 or start_index > end_index:
        return None

    xml_document = s[start_index : end_index + len(end_tag)]

    try:
        root = parse_document(xml_document)

    # This basically never happens now that we have recover=True on the parser
    except etree.XMLSyntaxError as e:
        raise ValueError("Invalid XML document: " + str(e))

    return root


def render_inner(node: etree._Element) -> str:
    """Render the inner text and markup of an XML node, minus the outer tags."""
    result = []

    # Some text may come before the first child
    if node.text:
        result.append(node.text)

    # Add each child's string representation
    for child in node:
        result.append(etree.tostring(child, encoding="unicode"))

    return "".join(result)


def parse_return_value(s: str) -> str:
    root = find_and_parse_xml(s, root_tag="return-value")
    if root is not None:
        json_str = render_inner(root)
        data_spec = json.loads(json_str)
    else:
        data_spec = json.loads(s)

    return data_spec
