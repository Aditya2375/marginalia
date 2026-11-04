# Marginalia

Local PDF/TXT/Markdown passage search, with extracted quotes and page/section citations. Similarity retrieval is approximate. It can miss an answer or retrieve unrelated text. Optional GPU-written answers can invent details; check every quote and citation.

Live: https://marginalia-eta-eight.vercel.app

## Limits and privacy

- Up to 10 MB, 120 PDF pages, 300,000 text characters, 40,000 words and 240 passages. Text PDFs only, no OCR.
- Document indexing has a three-minute deadline; questions three minutes; search model startup two minutes; embedding requests 45 seconds. Reset invalidates old results and terminates the search worker. Questions are capped at 500 characters, chat at ten recent exchanges.
- Browser processing is local. The app does not intentionally post document text or questions to a server. Third-party libraries/model assets download from jsDelivr, Hugging Face and model CDNs, which receive network metadata. Dependencies run code on the page. Do not use this demo for confidential or untrusted documents.
- Search model downloads and optional WebGPU writing can consume substantial bandwidth, memory and battery. Browser caches may retain downloaded models. Written answers have a two-minute UI deadline; reload the tab to stop all GPU work if a runtime ignores cancellation.
- PDF.js is pinned to 6.5.438 and PDF eval is disabled. Third-party assets are pinned where supported, not self-hosted or independently audited. No backend, account, paid API, database or server queue.

## Check locally

Serve this folder over HTTP, then open index.html. Worker/model downloads need internet. Browser tests should cover a small PDF, TXT, empty/scanned/corrupt files, reset during indexing, repeated uploads, literal HTML in text, question limits, citation preview, model-network failure and desktop/mobile layout.

Historical README wording overstated answer guarantees. Quote mode searches; it does not reason, calculate or verify factual truth.
