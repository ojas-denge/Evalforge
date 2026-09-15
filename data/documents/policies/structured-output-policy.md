\# Structured Output Policy



Structured LLM responses should conform to a defined application schema.



Schema validation should occur before downstream components rely on generated fields.



Required fields must be present and values must satisfy their declared constraints.



Validation errors should identify the failed field or constraint when possible.



A repair or retry operation may be used when configured, but the original invalid response should remain observable.

