---
tags: [project, software]
---
# Swim Log Schema

Data model for swims in the [[Lighthouse App]].

## Tables

- `swims`: id, beach_id, started_at, minutes, water_temp_c, notes
- `photos`: id, swim_id, blob, taken_at
- `beaches`: id, name, station_code

The station code links each beach to its row in the tide CSV.

## Migrations

Version 2 added `water_temp_c` as a nullable real column. Old swims keep a null temperature instead of a made-up zero. Photos are stored as blobs in IndexedDB, never inline in the swim row.
