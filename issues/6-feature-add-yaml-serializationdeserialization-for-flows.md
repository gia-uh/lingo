---
number: 6
title: "Feature: Add YAML serialization/deserialization for Flows"
state: open
labels:
---

**Description**

Implement a feature to serialize `Flow` objects to and from a YAML format.

**Problem**

Currently, `Flow` objects are defined exclusively in Python code. This is great for developers but makes it difficult to:
1.  Save, store, or version control complex, configured workflows.
2.  Allow non-developers (like domain experts or prompt engineers) to create, view, or edit these workflows.

**Proposed Solution**

Create a YAML schema that can represent the structure of a `Flow`, including its sequence of nodes (`system`, `reply`, `invoke`, `decide`, `choose`, `route`, etc.) and their parameters.

This would involve:
-   A `Flow.to_yaml()` method to serialize an existing `Flow` object.
-   A `Flow.from_yaml()` class method to deserialize a YAML file or string back into a `Flow` object.

This would enable `lingo` workflows to be managed as configuration files, allowing domain experts to write custom flows without needing to write Python code, and making the flows themselves more portable.
