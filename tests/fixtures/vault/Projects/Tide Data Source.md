---
tags: [project, data]
---
# Tide Data Source

Tide predictions for the Lighthouse App come from the harbour authority as a CSV, published on the first of each month. Each row has a station code, a timestamp in UTC and a height in metres.

## Gotchas

Timestamps are UTC, so the app must convert to local time and handle the clock change in March and October. Heights below chart datum show up as negative numbers and are valid.

The station code for the north beach is NB04. The south cove uses SC11.

Back to [[Lighthouse App#Decisions|the app decisions]].
