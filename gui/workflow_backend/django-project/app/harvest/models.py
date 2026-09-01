"""Local copy of the OAI-PMH repository, kept in sync by ``manage.py harvest_oai``."""

import uuid

from django.db import models


class HarvestedRecord(models.Model):
    """One OAI-PMH record (a repository folder), upserted by the harvester."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    oai_identifier = models.CharField(max_length=255, unique=True)
    # OAI datestamp kept verbatim (ISO-8601 UTC), so lexicographic order is
    # chronological and the value can be passed back as ``from=`` unchanged.
    datestamp = models.CharField(max_length=64, blank=True, default="")
    set_specs = models.JSONField(default=list, blank=True)
    deleted = models.BooleanField(default=False)
    metadata = models.JSONField(default=dict, blank=True)
    files = models.JSONField(default=list, blank=True)
    # Lower-cased concatenation of the searchable fields, built at write time
    # (services.build_search_text), so search is a substring match per term.
    search_text = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "harvested_records"
        ordering = ["-datestamp", "oai_identifier"]

    def __str__(self):
        return self.oai_identifier


class HarvestRun(models.Model):
    """History of harvester runs; the newest successful one carries the watermark."""

    class Status(models.TextChoices):
        SUCCESS = "success", "Success"
        ERROR = "error", "Error"

    class Mode(models.TextChoices):
        INCREMENTAL = "incremental", "Incremental"
        FULL = "full", "Full"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    status = models.CharField(max_length=16, choices=Status.choices)
    mode = models.CharField(max_length=16, choices=Mode.choices)
    from_datestamp = models.CharField(max_length=64, blank=True, default="")
    # Highest datestamp observed by a completed run; the next incremental run
    # resumes from here. Only successful runs advance it, so a failed run can
    # never skip past records it did not store.
    watermark = models.CharField(max_length=64, blank=True, default="")
    records_seen = models.IntegerField(default=0)
    records_deleted = models.IntegerField(default=0)
    error = models.TextField(blank=True, default="")
    started_at = models.DateTimeField()
    finished_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "harvest_runs"
        ordering = ["-finished_at"]

    def __str__(self):
        return f"HarvestRun {self.mode}/{self.status} @ {self.finished_at}"
