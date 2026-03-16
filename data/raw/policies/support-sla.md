# Support SLA Policy
Doc ID: policy-support-sla
Last updated: 2025-10-02

Severity levels
- P1 (major outage, data loss): acknowledge within 5 minutes, mitigation started within 15 minutes, customer updates every 30 minutes via status page + direct email to account owner.
- P2 (degraded core feature): acknowledge within 15 minutes, mitigation within 1 business hour, updates hourly.
- P3 (minor defect): acknowledge same business day, updates every 2 business days.

Communication rules
- Always create an incident record and tie customer tickets to it.
- During P1/P2, post updates simultaneously to the status page, shared Slack channels, and the ticket.
- Close incident only after a customer-visible RCA draft exists.
