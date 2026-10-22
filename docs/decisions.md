# Decisions

| # | Decision | Why |
|---|---|---|
| 1 | Static React 19 + TypeScript 7 + Vite 8, no backend | Brief: static client-side app; nothing needs a server. Versions pinned exactly in package.json / lockfile. |
| 2 | Custom SVG canvas instead of React Flow | Semantic zoom needs level-of-detail text tied to on-screen size, keyboard arrow navigation between nodes and a separate "open deeper level" action. A few hundred lines of SVG gave full control without a large dependency. |
| 3 | KaTeX for equations, zod for schemas | Small, mature; zod validates profiles, deep-link state and every loaded data file (samples, tokenizer, offsets, catalogs). |
| 4 | Hash routing (`#/explore?m=…&lv=…`) | Works on any static host (here.now, GitHub Pages) without server rewrites; deep links survive reload. SPA fallback is still enabled on here.now. |
| 5 | One shape table (`src/models/shapes.ts`) drives graph labels, inspector, catalog, treemap, memory lab, tests | Prevents drifting copies of numbers. Accounting throws if family sums and component sums disagree. |
| 6 | Two independent toy code paths: `forwardFull` (layer-major, explicit mask) and `decodeStep` (token-major with cache) | Makes the cache-equivalence check meaningful rather than tautological. |
| 7 | Training toy uses scalar reverse-mode autodiff plus a separate plain-number forward | Finite differences on the plain forward independently check the backward pass. |
| 8 | Toy weights derived per tensor name from the seed | Editing or regenerating one tensor never shifts the random stream of others; seeds are shareable. |
| 9 | Real values only from bounded HTTP Range reads (4.5 MB total of a 100 MB budget) | Brief's extraction policy. Every response checked for 206, exact Content-Range and length before acceptance. |
| 10 | Logical per-expert tensors are the counting unit; packed runtime tensors shown as a mapping | The checkpoint stores per-expert tensors; Transformers packs at load time. Counting both would double count. |
| 11 | Model cards' "non-embedding" = total − embedding − head | Only this convention reproduces 6.95B and 29.9B. Found during review; tests pin it. |
| 12 | Qwen3.8 "layer 2" displayed as config value 2 = code index 1 | Resolved from the implementation (`layer_idx + 1 in ple_layer_ids`) and the checkpoint tensor names. |
| 13 | Build id generated from git at build time into an untracked file | No self-referential commit hash; dirty trees are labelled. Deploy only builds made from a clean tree. |
| 14 | No runtime third-party requests | Fonts are system fonts; KaTeX fonts are bundled; site works offline after load. |
| 15 | GitHub repo private (brief default); here.now authenticated publish with saved credentials | Persistent deployment without exposing secrets. |
