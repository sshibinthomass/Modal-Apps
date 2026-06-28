# LLM Inference TypeScript

TypeScript clients for running authenticated inference against Modal-hosted LLM and image generation models. Each script loads credentials from the shared `.env` file at the repository root and calls a secured Modal FastAPI endpoint using proxy authentication headers.

---

## Prerequisites

- **Node.js** v18+ (v22 recommended)
- **npm** v9+
- A `.env` file at the repository root (`d:/Github-Projects/Modal-Apps/.env`) containing:
  ```
  MODAL_KEY=<your-modal-key>
  MODAL_SECRET=<your-modal-secret>
  ```

---

## Setup

```bash
cd llm-inference-typescript
npm install
```

---

## Available Scripts

| Script | Command | Model | Output |
|---|---|---|---|
| `start-gemma` | `npm run start-gemma` | Gemma 4 12B IT | Text printed to console |
| `start-llama` | `npm run start-llama` | Llama 3.2 3B Instruct | Text printed to console |
| `start-flux-dev` | `npm run start-flux-dev` | FLUX.1-dev | PNG image in `outputs/` |
| `start-flux-schnell` | `npm run start-flux-schnell` | FLUX.1-schnell | PNG image in `outputs/` |
| `start-trellis` | `npm run start-trellis` | TRELLIS 2-4B | GLB 3D asset in `outputs/` |
| `start-trellis-fast` | `npm run start-trellis-fast` | TRELLIS 2-4B Fast | GLB 3D asset in `outputs/` |

> **Note:** You can also run from the repository root using the `--prefix` flag:
> ```bash
> npm --prefix llm-inference-typescript run start-gemma
> ```

---

## Output File Naming

Generated output files are saved in `outputs/` with the following naming conventions:

| Model | Pattern |
|---|---|
| FLUX.1-dev | `flux_dev_YYYYMMDD_HHmmss.png` |
| FLUX.1-schnell | `flux_schnell_YYYYMMDD_HHmmss.png` |
| TRELLIS 2-4B | `<input_stem>_trellis_YYYYMMDD_HHmmss.glb` |
| TRELLIS 2-4B Fast | `<input_stem>_trellis_fast_YYYYMMDD_HHmmss.glb` |

---

## Source Files

```
src/
├── inf-gemma-4-12b-it.ts         # Gemma 4 12B IT text inference
├── inf-llama-3-2-3b-instruct.ts  # Llama 3.2 3B Instruct text inference
├── inf-flux-dev.ts               # FLUX.1-dev text-to-image
├── inf-flux-schnell.ts           # FLUX.1-schnell text-to-image
├── inf-trellis-2-4b.ts           # TRELLIS 2-4B image-to-3D
└── inf-trellis-2-4b-fast.ts      # TRELLIS 2-4B Fast image-to-3D
```

---

## Authentication

All endpoints require **Modal proxy authentication** via HTTP headers:

```
Modal-Key: <value from MODAL_KEY>
Modal-Secret: <value from MODAL_SECRET>
```

These are loaded automatically from the root `.env` file. Requests without valid credentials will receive a `401 Unauthorized` response.

---

## Project Structure

```
llm-inference-typescript/
├── src/                 # TypeScript source files (one per model)
├── outputs/             # Generated images and 3D assets (git-ignored)
├── dist/                # Compiled JS output (git-ignored)
├── package.json         # Scripts and dependencies
└── tsconfig.json        # TypeScript compiler config
```
