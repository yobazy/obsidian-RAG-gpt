---
tags: [project, software]
status: active
started: 2026-01-15
---
# Lighthouse App

A small offline-first app for logging tide times and sea swims along the coast. Built with SvelteKit and SQLite in the browser.

## Goals

Show the next two high tides for a chosen beach, log each swim with water temperature, and work with no signal on the cliffs.

## Decisions

- Picked IndexedDB over localStorage because swim logs with photos went past the 5 MB limit in testing.
- Tide data comes from a monthly CSV download, not a live API, so it works offline. See [[Tide Data Source]].
- Dropped push notifications for version one. Too fiddly on iOS for the value.

## Next steps

Write the import script for the CSV, then test on the [[Coast Path Ride]] weekend where signal is poor.
