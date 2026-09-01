"""Harvest the configured OAI-PMH repository into the local database.

One run per invocation; the compose ``harvester`` service loops it. Runs are
all-or-nothing: on an upstream error nothing is stored and the watermark does
not advance, so the next run retries the same window.
"""

import os

from app.harvest import services
from app.harvest.models import HarvestedRecord, HarvestRun
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone

KEPT_RUNS = 200


class Command(BaseCommand):
    help = "Harvest OAI-PMH metadata records into the harvested_records table."

    def add_arguments(self, parser):
        parser.add_argument(
            "--full",
            action="store_true",
            help=(
                "Re-harvest everything (ignore the watermark) and mark records "
                "missing upstream as deleted."
            ),
        )
        parser.add_argument(
            "--max-records",
            type=int,
            default=10000,
            help="Safety cap on records fetched in one run.",
        )

    def handle(self, *args, **options):
        if not os.environ.get("OAI_PMH_BASE_URL", "").rstrip("/"):
            # Exit 0 so the compose loop stays quiet on unconfigured deployments.
            self.stdout.write("OAI_PMH_BASE_URL is not set; harvesting disabled.")
            return
        started_at = timezone.now()
        full = options["full"]
        mode = HarvestRun.Mode.FULL if full else HarvestRun.Mode.INCREMENTAL
        last = None if full else services.latest_success_run()
        from_date = last.watermark if last else ""

        envelope = services.make_client().list_records(
            metadata_prefix="mdrs",
            from_date=from_date,
            max_records=options["max_records"],
        )
        if envelope["status"] == "error":
            HarvestRun.objects.create(
                status=HarvestRun.Status.ERROR,
                mode=mode,
                from_datestamp=from_date,
                error=f"{envelope['error_code']}: {envelope['error']}",
                started_at=started_at,
            )
            raise CommandError(f"Harvest failed: {envelope['error']}")

        records = envelope["records"]
        # ``from`` is inclusive, so the boundary record is re-fetched next run
        # and absorbed by the upsert; the max() keeps the watermark monotonic.
        watermark = max([from_date] + [r.get("datestamp", "") for r in records])
        deleted_count = sum(1 for r in records if r.get("deleted"))
        with transaction.atomic():
            services.upsert_records(records)
            if full:
                seen = {r.get("identifier") for r in records}
                deleted_count += (
                    HarvestedRecord.objects.filter(deleted=False)
                    .exclude(oai_identifier__in=seen)
                    .update(deleted=True)
                )
            HarvestRun.objects.create(
                status=HarvestRun.Status.SUCCESS,
                mode=mode,
                from_datestamp=from_date,
                watermark=watermark,
                records_seen=len(records),
                records_deleted=deleted_count,
                started_at=started_at,
            )
            keep = HarvestRun.objects.order_by("-finished_at").values_list(
                "pk", flat=True
            )[:KEPT_RUNS]
            HarvestRun.objects.exclude(pk__in=list(keep)).delete()
        self.stdout.write(
            self.style.SUCCESS(
                f"Harvested {len(records)} records "
                f"(from={from_date or 'epoch'}, watermark={watermark or 'none'})"
            )
        )
