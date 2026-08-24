"""Serve the OpenAPI spec and a Swagger UI page for it.

Deliberately dependency-free. `drf-spectacular` would generate a schema from the
serializers, but this API's serializers are plain `Serializer` subclasses with no
type hints to introspect, so a generated schema would be mostly empty objects —
and adding the package means a ~20 minute image rebuild for a one-endpoint API.

The spec is hand-authored in `openapi.yaml` at the project root. These two views
just publish it.
"""

from pathlib import Path

from django.conf import settings
from django.http import FileResponse, HttpResponse, HttpResponseNotFound

SCHEMA_PATH = Path(settings.BASE_DIR) / "openapi.yaml"

# Pinned rather than @latest: a silent major bump in the CDN would change the
# rendering of a page that is meant to be stable documentation.
SWAGGER_UI_VERSION = "5.17.14"

_SWAGGER_HTML = f"""<!DOCTYPE html>
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
    #offline {{
      display: none; margin: 3rem auto; max-width: 40rem; padding: 1.5rem 2rem;
      font: 15px/1.6 system-ui, -apple-system, Segoe UI, sans-serif;
      border: 1px solid #d0d7de; border-radius: 8px; background: #fff;
    }}
    #offline code {{ background: #f3f4f6; padding: .1em .35em; border-radius: 4px; }}
  </style>
</head>
<body>
  <div id="swagger-ui"></div>

  <!-- Swagger UI is loaded from a CDN, so this page needs internet access even
       though the API itself does not. If the script fails, say so plainly
       rather than showing an empty page. -->
  <div id="offline">
    <h2>Swagger UI could not be loaded</h2>
    <p>This page fetches Swagger UI from a CDN and the request failed — most
       likely no internet access.</p>
    <p>The API itself is unaffected. The raw spec is always available at
       <code>/api/schema/</code>; download it and open it in
       <a href="https://editor.swagger.io">editor.swagger.io</a>, Postman, or any
       OpenAPI tool.</p>
  </div>

  <script src="https://unpkg.com/swagger-ui-dist@{SWAGGER_UI_VERSION}/swagger-ui-bundle.js"
          onerror="document.getElementById('offline').style.display='block'"></script>
  <script>
    window.addEventListener('load', function () {{
      if (typeof SwaggerUIBundle === 'undefined') {{
        document.getElementById('offline').style.display = 'block';
        return;
      }}
      SwaggerUIBundle({{
        url: '/api/schema/',
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


def openapi_schema(request):
    """Serve the raw spec, for Swagger UI and for any other OpenAPI tool."""
    if not SCHEMA_PATH.is_file():
        return HttpResponseNotFound(
            f"openapi.yaml not found at {SCHEMA_PATH}. It is expected at the "
            "project root."
        )
    # `application/yaml` is the registered type (RFC 9512). Swagger UI parses
    # YAML regardless of what it is served as.
    return FileResponse(SCHEMA_PATH.open("rb"), content_type="application/yaml")


def swagger_ui(request):
    """Render the interactive API docs."""
    return HttpResponse(_SWAGGER_HTML, content_type="text/html; charset=utf-8")
