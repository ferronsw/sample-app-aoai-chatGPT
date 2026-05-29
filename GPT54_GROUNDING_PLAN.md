# GPT-5.4 Grounding Plan

## Context

This repo currently grounds chat responses through classic Azure OpenAI On Your Data by sending `data_sources` in the Azure OpenAI request path. That path is no longer a viable target for GPT-5.4:

- Classic Azure OpenAI On Your Data is deprecated.
- Classic On Your Data is documented as supporting GPT-4o and GPT-4o-mini, not GPT-5.4.
- GPT-5.4 requires newer model APIs and should not be coupled to the legacy `data_sources` flow in this app.

The goal is to preserve Azure AI Search grounding while enabling GPT-5.4.

## Decision Summary

There are two realistic paths:

1. Best long-term managed architecture: Azure AI Search agentic retrieval / Foundry IQ.
2. Best near-term engineering choice: app-managed Azure AI Search retrieval plus GPT-5.4.

## Recommendation

Use option 2 first if the goal is to get GPT-5.4 working in this repo with minimal product disruption.

Use option 1 if the goal is to align with Microsoft’s forward-looking managed RAG architecture, improve retrieval quality over time, and accept a larger architectural migration.

## Option 1: Azure AI Search Agentic Retrieval / Foundry IQ

### Outcome

Move grounding ownership out of this app’s custom retrieval code and out of classic Azure OpenAI On Your Data. Use Azure AI Search agentic retrieval, ideally through Foundry IQ or a Foundry agent architecture, as the managed grounding layer for GPT-5.4.

### Why choose this

- Best alignment with Microsoft’s current long-term direction for grounded AI apps.
- Better support for conversational and complex retrieval through query planning and multi-query execution.
- Managed structured retrieval results, citations, and retrieval metadata.
- Better fit for future agent scenarios and richer orchestration.

### Tradeoffs

- Larger architecture change than a direct repo refactor.
- More Azure-side setup: knowledge sources, knowledge base, retrieval actions, and possibly agent infrastructure.
- Potentially more moving parts for local development and CI.
- Higher migration cost if the current app must keep its exact API shape and hosting model.

### Target architecture

1. Azure AI Search becomes the managed knowledge layer.
2. Foundry IQ or agentic retrieval performs retrieval planning and grounding.
3. The app calls the managed retrieval layer or agent endpoint.
4. GPT-5.4 generates answers using the structured grounding payload.
5. The app adapts the returned citations into the frontend’s existing response shape.

### Required workstreams

#### 1. Azure resource and data design

1. Confirm the existing Azure AI Search index is compatible with agentic retrieval requirements.
2. Review chunking, metadata fields, ACL fields, title/filepath/url coverage, and embedding strategy.
3. Decide whether to reuse the current index or create a dedicated knowledge base pipeline.
4. Define the retrieval contract: citations, document metadata, security trimming, query diagnostics.

#### 2. Managed retrieval configuration

1. Set up knowledge sources and knowledge base objects.
2. Configure retrieval instructions and result shaping.
3. Map security requirements to supported access-control patterns.
4. Validate citation quality and result determinism for the app’s use cases.

#### 3. Application integration

1. Replace the current `data_sources` request path in [app.py](./app.py).
2. Introduce a retrieval adapter layer that calls the managed retrieval endpoint.
3. Normalize managed grounding results into the frontend’s existing `tool` message and citation format.
4. Update title generation and history flows so they do not depend on legacy On Your Data response fields.

#### 4. GPT-5.4 model integration

1. Move Azure OpenAI calls to the v1 client path.
2. Prefer the Responses API unless there is a hard requirement to preserve chat completions.
3. Remove or gate unsupported reasoning-model parameters.
4. Validate output shape, streaming behavior, and citation attachment.

#### 5. Operational readiness

1. Update deployment configuration and environment-variable documentation.
2. Add observability for retrieval latency, empty-result rates, and answer grounding quality.
3. Add regression tests for grounded responses and unsupported-configuration failures.

### Phased delivery plan

#### Phase 0: Discovery

1. Inventory the current Azure AI Search schema and metadata quality.
2. Confirm supported agentic retrieval primitives for the target Azure region and subscription.
3. Decide whether Foundry IQ or direct agentic retrieval is the primary managed interface.

#### Phase 1: Proof of concept

1. Create a minimal retrieval-backed GPT-5.4 flow outside the current chat history path.
2. Validate citation fidelity and grounding quality on representative queries.
3. Measure latency and failure modes.

#### Phase 2: App integration

1. Replace the existing grounding path in the conversation endpoint.
2. Preserve the frontend response contract.
3. Add telemetry and test coverage.

#### Phase 3: Migration and cleanup

1. Remove legacy `data_sources`-based logic.
2. Update docs and deployment defaults.
3. Retire classic On Your Data assumptions from settings and validation paths.

### Validation criteria

1. GPT-5.4 returns grounded answers with citations for the top user journeys.
2. No `data_sources` payload is sent to Azure OpenAI.
3. Citation links, file paths, and titles render correctly in the existing UI.
4. Security trimming still applies to retrieval results.
5. Latency remains acceptable for chat UX.

### Risks

1. Managed retrieval features may introduce extra Azure setup and deployment complexity.
2. The current app’s UI contract may not map one-to-one with managed retrieval metadata.
3. Streaming behavior may differ from the current chat completions flow.
4. Regional support or preview dependencies may constrain rollout timing.

## Option 2: App-Managed Azure AI Search Retrieval Plus GPT-5.4

### Outcome

Keep Azure AI Search as the grounding source, but move retrieval fully into this app. The backend queries Azure AI Search directly, selects grounding chunks, and then calls GPT-5.4 as a standard model request without `data_sources`.

### Why choose this

- Fastest path to GPT-5.4 while preserving the current app and frontend contract.
- Keeps control in the application layer.
- Reuses the existing Azure AI Search index and most of the current configuration.
- Lower migration risk than a full managed retrieval architecture.

### Tradeoffs

- The app now owns retrieval orchestration, citation shaping, and relevance tuning.
- More custom code to maintain than a managed retrieval layer.
- Retrieval quality depends on the app’s query logic unless more advanced planning is added later.

### Target architecture

1. The backend receives the user message and history.
2. The backend runs Azure AI Search directly using the configured query mode.
3. The backend applies ACL trimming and selects top chunks.
4. The backend builds a grounded GPT-5.4 prompt or Responses API input from those chunks.
5. The backend formats Azure AI Search results as citations and returns them in the current UI contract.

### Required code changes

#### 1. Retrieval layer

1. Add a backend retrieval module for Azure AI Search runtime queries.
2. Support the current search modes where practical:
   - simple
   - semantic
   - vector
   - vector hybrid
3. Reuse existing settings in [backend/settings.py](./backend/settings.py) for index names, semantic config, vector fields, and security filters.
4. Reuse or move the group-based filter logic currently in [backend/utils.py](./backend/utils.py).

#### 2. Model call path

1. Remove the `extra_body.data_sources` path from grounded GPT-5.4 requests in [app.py](./app.py).
2. Build the model input from retrieved chunks instead.
3. Move the Azure OpenAI client path toward v1 usage.
4. Prefer the Responses API for GPT-5.4 unless preserving the existing chat completions streaming path is a hard requirement.
5. Gate reasoning-incompatible parameters by model family.

#### 3. Citation shaping

1. Build citations from Azure AI Search documents rather than model-supplied `message.context`.
2. Preserve the frontend-compatible citation structure already defined in [frontend/src/api/models.ts](./frontend/src/api/models.ts).
3. Keep the current `tool` message pattern if that minimizes frontend change.

#### 4. History and title generation

1. Ensure chat history writes still work with the new grounded response shape.
2. Keep title generation on the plain model path without retrieval.
3. Ensure the `/history/update` flow still stores tool and assistant messages correctly.

#### 5. Configuration and docs

1. Redefine `DATASOURCE_TYPE=AzureCognitiveSearch` to mean app-managed grounding, not classic On Your Data.
2. Update README and sample env docs to reflect the new behavior.
3. Clarify GPT-5.4 support expectations and any required embedding/search prerequisites.

### Phased delivery plan

#### Phase 0: Design

1. Confirm which Azure AI Search query modes must be supported on day one.
2. Decide whether to keep the current response shape exactly or allow small backend-only normalization.
3. Decide whether the first GPT-5.4 version uses Chat Completions or Responses API.

#### Phase 1: Minimum viable grounding

1. Add direct Azure AI Search lookup for `AzureCognitiveSearch`.
2. Build grounded prompt assembly for GPT-5.4.
3. Return citations in the existing frontend-compatible structure.
4. Disable legacy `data_sources` for this path.

#### Phase 2: Retrieval quality

1. Add hybrid retrieval if vectors exist.
2. Add semantic ranking and better result selection.
3. Improve chunk formatting, de-duplication, and prompt budget management.
4. Add telemetry around retrieval quality and latency.

#### Phase 3: Client modernization

1. Move to Azure OpenAI v1 client usage.
2. Migrate grounded generation to the Responses API if not already done.
3. Remove preview-version branching that only exists for classic On Your Data.

### Best-practice retrieval rules for this option

1. Use hybrid retrieval by default when vector fields are available.
2. Use semantic ranking where supported.
3. Keep retrieved content chunked and concise.
4. Return only the fields required for grounding and citations.
5. Apply security trimming at query time, not after generation.
6. Cap the grounding payload to a predictable token budget.
7. Log retrieval diagnostics separately from model-generation diagnostics.

### Validation criteria

1. GPT-5.4 grounded answers work with the existing Azure AI Search index.
2. Citations render correctly without relying on legacy `message.context` from Azure OpenAI.
3. Existing chat history flows still work.
4. Search ACL filtering still works.
5. Unit and integration tests cover retrieval, grounding, and failure cases.

### Risks

1. Retrieval quality may initially regress if the first version uses simpler query logic than the managed path.
2. The app will carry more orchestration logic and testing burden.
3. Response streaming may need careful handling if citations are only available after retrieval completes.

## Comparison

| Area | Option 1: Agentic Retrieval / Foundry IQ | Option 2: App-Managed Retrieval |
| --- | --- | --- |
| Time to first GPT-5.4 result | Slower | Faster |
| Architectural change | Larger | Moderate |
| Managed grounding capabilities | Strongest | App-owned |
| Fit for current repo | Lower | Higher |
| Citation and retrieval metadata | Managed | Custom-built |
| Long-term alignment | Best | Good but custom |

## Suggested Sequence

If engineering time is limited, do this:

1. Implement option 2 to unblock GPT-5.4 with Azure AI Search.
2. Stabilize retrieval quality, citations, telemetry, and deployment.
3. Reassess whether option 1 is worth the larger migration once the product path is validated.

If platform modernization is the top priority, do this:

1. Run a small proof of concept for option 1.
2. Compare retrieval quality and latency against option 2.
3. Commit only if the managed benefits justify the migration overhead.

## Immediate Next Steps

### If choosing option 1

1. Verify Azure region and feature readiness for agentic retrieval / Foundry IQ.
2. Inventory the current index schema and metadata quality.
3. Build a retrieval proof of concept before touching this repo’s runtime path.

### If choosing option 2

1. Add a direct Azure AI Search runtime retrieval module.
2. Replace the legacy `data_sources` path in [app.py](./app.py).
3. Preserve the current frontend citation contract.
4. Modernize the Azure OpenAI client path after the first grounded version works.