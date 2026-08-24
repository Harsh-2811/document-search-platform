"""Fetch a shared PDF from Google Drive into data/.

    python manage.py fetch_drive https://drive.google.com/file/d/<FILE_ID>/view
    python manage.py fetch_drive <FILE_ID> --name resume

The filename comes from Drive itself (its `Content-Disposition` header), so
pasting the share link is all that's needed. `--name` overrides it.

Downloading is separate from ingesting on purpose: fetch, eyeball the PDF,
then run `ingest_docs`. Chaining them would mean a bad download silently
becomes bad chunks.

The transport lives in `rag/drive_download.py`, which is Django-free; this
command only maps arguments to it and turns failures into `CommandError`.
"""

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

# Always data/ — that is where `ingest_docs` looks by default, so downloading
# anywhere else would just mean passing an explicit path to the next command.
DEST_DIR = Path("data")


class Command(BaseCommand):
    help = "Download a publicly-shared Google Drive PDF into data/."

    def add_arguments(self, parser):
        parser.add_argument(
            "url",
            help=(
                "Google Drive share link, or a bare file id. The file must be "
                'shared as "Anyone with the link".'
            ),
        )
        parser.add_argument(
            "--name",
            default=None,
            help=(
                "Save as <name>.pdf instead of the name Drive reports. "
                "The .pdf extension is added if you leave it off."
            ),
        )
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Replace the file if it already exists.",
        )

    def handle(self, *args, **options):
        # Imported here so `manage.py help` doesn't pay for requests, and so a
        # missing dependency surfaces when the command runs, not at startup.
        from rag.drive_download import DriveDownloadError, download_drive_pdf

        self.stdout.write("Downloading from Google Drive...")
        try:
            path = download_drive_pdf(
                options["url"],
                DEST_DIR,
                name=options["name"],
                overwrite=options["overwrite"],
            )
        except DriveDownloadError as exc:
            raise CommandError(str(exc)) from exc

        size_kb = path.stat().st_size / 1024
        self.stdout.write(self.style.SUCCESS(f"Saved {path} ({size_kb:.0f} KB)"))
        self.stdout.write(f"Next: python manage.py ingest_docs {path}")
