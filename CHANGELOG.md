## v0.4.1 (2024-12-05)

### Fix

- add and enable OpenAI strict mode (#55)
- support embedding with LiteLLM for Ragas (#56)

## v0.4.0 (2024-12-04)

### Feat

- improve late chunking and optimize pgvector settings (#51)

## v0.3.0 (2024-12-03)

### Feat

- support prompt caching and apply Anthropic's long-context prompt format (#52)

## v0.2.1 (2024-11-22)

### Fix

- improve structured output extraction and query adapter updates (#34)
- upgrade rerankers and remove flashrank patch (#47)
- improve unpacking of keyword search results (#46)
- add fallbacks for model info (#44)

## v0.2.0 (2024-10-21)

### Feat

- add Chainlit frontend (#33)

## v0.1.4 (2024-10-15)

### Fix

- fix optimal chunking edge cases (#32)

## v0.1.3 (2024-10-13)

### Fix

- upgrade pdftext (#30)
- improve chunk and segment ordering (#29)

## v0.1.2 (2024-10-08)

### Fix

- avoid pdftext v0.3.11 (#27)

## v0.1.1 (2024-10-07)

### Fix

- patch rerankers flashrank issue (#22)

## v0.1.0 (2024-10-07)

### Feat

- add reranking (#20)
- add LiteLLM and late chunking (#19)
- add PostgreSQL support (#18)
- make query adapter minimally invasive (#16)
- upgrade default CPU model to Phi-3.5-mini (#15)
- add evaluation (#14)
- infer missing font sizes (#12)
- automatically adjust number of RAG contexts (#10)
- improve exception feedback for extraction (#9)
- optimize config for CPU and GPU (#7)
- simplify document insertion (#6)
- implement basic features (#2)
- initial commit

### Fix

- lazily import optional dependencies (#11)
- improve indexing of multiple documents (#8)
