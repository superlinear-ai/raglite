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
