"""Serve a Swagger UI page for the hand-authored OpenAPI spec.

Deliberately dependency-free. `drf-spectacular` would generate a schema from the
serializers, but this API's serializers are plain `Serializer` subclasses with no
type hints to introspect, so a generated schema would be mostly empty objects —
and adding the package means a ~20 minute image rebuild for a one-endpoint API.

The spec is hand-authored in `openapi.yaml` at the project root and **inlined
into the page** rather than fetched over HTTP. That keeps the API surface to the
one endpoint it actually serves: there is no schema route to secure, version, or
keep in sync with the docs page.
"""

import json
from pathlib import Path

import yaml
from django.conf import settings
from django.http import HttpResponse

SCHEMA_PATH = Path(settings.BASE_DIR) / "openapi.yaml"

# Pinned rather than @latest: a silent major bump in the CDN would change the
# rendering of a page that is meant to be stable documentation.
SWAGGER_UI_VERSION = "5.17.14"


def _spec_as_json() -> str:
    """Load openapi.yaml and render it as JSON safe to embed in a <script>.

    Returns "null" if the file is missing, which the page reports rather than
    rendering blank.
    """
    if not SCHEMA_PATH.is_file():
        return "null"

    spec = yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))
    # Escape `<` so a literal "</script>" anywhere in the spec text cannot close
    # the surrounding tag. JSON treats < as an ordinary character.
    return json.dumps(spec).replace("<", "\\u003c")


def swagger_ui(request):
    """Render the interactive API docs with the spec inlined."""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Document Search Platform — API</title>
  <link rel="stylesheet"
        href="https://unpkg.com/swagger-ui-dist@{SWAGGER_UI_VERSION}/swagger-ui.css">
  <style>
    body {{ margin: 0; background: #fafafa; }}
    .swagger-ui .topbar {{ display: none; }}
    #problem {{
      display: none; margin: 3rem auto; max-width: 40rem; padding: 1.5rem 2rem;
      font: 15px/1.6 system-ui, -apple-system, Segoe UI, sans-serif;
      border: 1px solid #d0d7de; border-radius: 8px; background: #fff;
    }}
    #problem code {{ background: #f3f4f6; padding: .1em .35em; border-radius: 4px; }}
  </style>
</head>
<body>
  <div id="swagger-ui"></div>

  <!-- Swagger UI is loaded from a CDN, so this page needs internet access even
       though the API itself does not. If the script fails, say so plainly
       rather than showing an empty page. -->
  <div id="problem">
    <h2>API docs could not be rendered</h2>
    <p id="problem-detail"></p>
    <p>The API itself is unaffected. The specification lives in
       <code>openapi.yaml</code> at the project root — open it in
       <a href="https://editor.swagger.io">editor.swagger.io</a>, Postman, or any
       OpenAPI tool.</p>
  </div>

  <script src="https://unpkg.com/swagger-ui-dist@{SWAGGER_UI_VERSION}/swagger-ui-bundle.js"
          onerror="window.__swaggerCdnFailed = true"></script>
  <script>
    const spec = {_spec_as_json()};

    function problem(message) {{
      document.getElementById('problem-detail').textContent = message;
      document.getElementById('problem').style.display = 'block';
    }}

    window.addEventListener('load', function () {{
      if (window.__swaggerCdnFailed || typeof SwaggerUIBundle === 'undefined') {{
        problem('Swagger UI could not be fetched from the CDN — most likely no '
              + 'internet access.');
        return;
      }}
      if (spec === null) {{
        problem('openapi.yaml was not found on the server.');
        return;
      }}
      SwaggerUIBundle({{
        spec: spec,
        dom_id: '#swagger-ui',
        deepLinking: true,
        displayRequestDuration: true,
        defaultModelsExpandDepth: 1,
        tryItOutEnabled: true
      }});
    }});
  </script>
</body>
</html>
"""
    return HttpResponse(html, content_type="text/html; charset=utf-8")
