# Changelog

## v0.1.0 — 2026-10-06

The first DecisionFlow release introduces:

- a declarative YAML/JSON workflow format for typed, multi-step decisions;
- provider-neutral adapters for local models, hosted APIs, and cached scores;
- a compiled structured decision-flow representation;
- greedy, A*, beam-search, dynamic-programming, SAT, enumeration,
  probabilistic-circuit, top-k, and sampling inference backends;
- joint MAP, marginal, valid-mass, and complete-flow outputs when supported by
  the selected backend;
- a service-ticket quickstart, CLI, agent-tool interface, and reusable Python
  API; and
- a separate experiment package covering the current benchmark pilots.
