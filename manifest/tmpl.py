from pathlib import Path

import jinja2

THIS_DIR = Path(__file__).parent
_TMPL_ENV = jinja2.Environment(
    loader=jinja2.FileSystemLoader(THIS_DIR / "templates"),
    undefined=jinja2.StrictUndefined,
)


def load(name: str) -> jinja2.Template:
    return _TMPL_ENV.get_template(name)
