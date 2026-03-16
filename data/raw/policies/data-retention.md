# Data Retention and Deletion
Doc ID: policy-data-retention
Revision: 2025-08-30

- Soft-deleted customer data stays in warm storage for 30 days to enable account recovery.
- Backups contain deleted data for up to 35 days; after that, purge jobs remove it from object storage and search indexes.
- Legal holds pause deletion timers; requires ticket tag `legal-hold` and director approval.
- Audit logs are retained for 13 months for compliance and are never deleted retroactively.
