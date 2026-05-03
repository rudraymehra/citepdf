# PDF-Constrained Chat

This Chainlit app talks to the FastAPI backend at `$API_BASE_URL` (default `http://localhost:8000`).

**How grounding works**:

1. We retrieve hybrid (dense + sparse) from a Qdrant index over your PDF.
2. **Deep mode** also runs HyDE + multi-query + step-back + decomposition, fuses with RRF, and adds a two-pass verifier that drops unsupported claims.
3. Every claim is bound to a citable chunk via the Anthropic Citations API. Citations show up in the side panel.
4. If the answer isn't in the document, you'll see the exact refusal line — never a hallucinated guess.

**Switch modes** via the chat-profile picker at the top: `instant` (~3 s) or `deep` (~30 s, strongest grounding).
