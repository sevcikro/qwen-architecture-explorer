// Shared by the live-demo page and its worker (kept separate so the page does not pull in Transformers.js).
export const MODEL_ID = 'onnx-community/Qwen3-0.6B-ONNX';
export const MODEL_REVISION = '58ca03a5e7b354b480b1a82b34bde8dc7b854cc0';
export const MODEL_URL = `https://huggingface.co/${MODEL_ID}/tree/${MODEL_REVISION}`;
/** Bytes of the ONNX weight file each variant downloads (from the Hub API at the pinned revision). */
export const VARIANT_BYTES = { q4f16: 569_789_750, q4: 919_096_585, q8: 617_687_575 } as const;
export type Dtype = keyof typeof VARIANT_BYTES;
