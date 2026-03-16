# Incident Response Runbook
Doc ID: ops-incident-runbook
Version: 2025-11-04

Scope: applies to P1 and P2 production incidents.

Steps
1) Acknowledge the alert within 5 minutes for P1 (15 minutes for P2).
2) Create or link to the central incident record and add affected customers.
3) Kick off mitigation within 15 minutes for P1; within 60 minutes for P2.
4) Customer comms: update every 30 minutes for P1, every 60 minutes for P2. Channels: status page, ticket, shared Slack.
5) Assign a comms lead if >60 minutes total impact.
6) Capture timelines and decisions in the incident doc; publish draft RCA before closing.

Mitigation templates
- CDN degradation: bypass cache, enable origin shielding, and lower TTL.
- Invoice export latency: warm cache by pre-loading invoice details; offer CSV export as workaround.
