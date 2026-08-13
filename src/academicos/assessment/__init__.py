"""Assessment Designer: Module 1 of AssessmentOS.

Blueprint -> question retrieval -> validation -> paper generation -> PDF export.
Deterministic core (chapter tagging, selection, optimizer, formatting); no LLM
calls in the critical path, matching the rest of academicos's design principles.
"""
