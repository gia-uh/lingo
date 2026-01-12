---
number: 8
title: "feat(flow): make Decide and Choose execution atomic"
state: closed
labels:
---

Branching nodes (Decide, Choose) now execute sub-flows in an isolated context using context.clone(). If a sub-flow succeeds, its message history is committed to the parent context. If it fails, the parent context remains unchanged, ensuring conversation integrity.

This prevents partial branch execution from polluting the main context with messages from logically failed paths.

Adds unit tests to verify atomicity for both success and failure cases.