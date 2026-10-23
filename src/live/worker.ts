/// <reference lib="webworker" />
// Runs Qwen3-0.6B (ONNX conversion) in the browser with Transformers.js, off the main thread.
// The ONNX runtime is served by this site (/ort/); only the model files are fetched from huggingface.co,
// and only after the visitor explicitly starts the demo.
import { AutoModelForCausalLM, AutoTokenizer, InterruptableStoppingCriteria, LogitsProcessor, LogitsProcessorList, TextStreamer, env, type PreTrainedModel, type PreTrainedTokenizer, type Tensor } from '@huggingface/transformers';

import { MODEL_ID, MODEL_REVISION, type Dtype } from './config';

export type ToWorker =
  | { type: 'load'; device: 'webgpu' | 'wasm'; dtype: Dtype }
  | { type: 'chat'; messages: { role: 'user' | 'assistant' | 'system'; content: string }[]; maxNewTokens: number; sample: boolean; temperature: number }
  | { type: 'next'; text: string; k: number }
  | { type: 'stop' };

export type FromWorker =
  | { type: 'progress'; file: string; loaded: number; total: number }
  | { type: 'ready'; device: string; dtype: string; seconds: number }
  | { type: 'token'; text: string; count: number; elapsed: number }
  | { type: 'done'; text: string; count: number; elapsed: number; promptTokens: number; stopped: 'end' | 'limit' | 'user' }
  | { type: 'next'; promptIds: number[]; pieces: string[]; top: { piece: string; id: number; p: number }[]; elapsed: number }
  | { type: 'error'; message: string };

const ctx = self as unknown as DedicatedWorkerGlobalScope;
const post = (m: FromWorker) => ctx.postMessage(m);

env.allowLocalModels = false;
env.backends.onnx.wasm!.wasmPaths = { mjs: new URL('/ort/ort-wasm-simd-threaded.asyncify.mjs', ctx.location.origin).href, wasm: new URL('/ort/ort-wasm-simd-threaded.asyncify.wasm', ctx.location.origin).href };
if (!ctx.crossOriginIsolated) env.backends.onnx.wasm!.numThreads = 1;

let tokenizer: PreTrainedTokenizer | null = null;
let model: PreTrainedModel | null = null;
const stopper = new InterruptableStoppingCriteria();

function toFloat32(t: Tensor): Float32Array {
  const d = t.data as unknown;
  if (d instanceof Float32Array) return d;
  if (d instanceof Uint16Array) {
    // IEEE half floats stored as raw bits
    const out = new Float32Array(d.length);
    for (let i = 0; i < d.length; i++) {
      const h = d[i], s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
      out[i] = e === 0 ? s * 2 ** -14 * (f / 1024) : e === 31 ? (f ? NaN : s * Infinity) : s * 2 ** (e - 15) * (1 + f / 1024);
    }
    return out;
  }
  return Float32Array.from(d as ArrayLike<number>);
}

ctx.onmessage = async (ev: MessageEvent<ToWorker>) => {
  const m = ev.data;
  try {
    if (m.type === 'load') {
      const t0 = performance.now();
      const progress_callback = (p: { status: string; file?: string; loaded?: number; total?: number }) => {
        if (p.status === 'progress' && p.file) post({ type: 'progress', file: p.file, loaded: p.loaded ?? 0, total: p.total ?? 0 });
      };
      tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, { revision: MODEL_REVISION, progress_callback });
      model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, { revision: MODEL_REVISION, dtype: m.dtype, device: m.device, progress_callback });
      post({ type: 'ready', device: m.device, dtype: m.dtype, seconds: (performance.now() - t0) / 1000 });
    } else if (m.type === 'stop') {
      stopper.interrupt();
    } else if (m.type === 'chat') {
      if (!tokenizer || !model) throw new Error('model not loaded');
      const inputs = tokenizer.apply_chat_template(m.messages, { add_generation_prompt: true, return_dict: true, enable_thinking: false } as never) as unknown as { input_ids: Tensor };
      const promptTokens = Number(inputs.input_ids.dims[1]);
      let text = '', count = 0;
      const t0 = performance.now();
      const streamer = new TextStreamer(tokenizer, {
        skip_prompt: true, skip_special_tokens: true,
        callback_function: (piece: string) => { text += piece; post({ type: 'token', text, count, elapsed: (performance.now() - t0) / 1000 }); },
        token_callback_function: () => { count++; },
      });
      stopper.reset();
      const out = await model.generate({ ...inputs, max_new_tokens: m.maxNewTokens, do_sample: m.sample, temperature: m.temperature, top_k: 20, top_p: 0.8, streamer, stopping_criteria: stopper } as never) as Tensor;
      const ids = Array.from(out.tolist()[0] as bigint[]).slice(promptTokens).map(Number);
      const endId = tokenizer.convert_tokens_to_ids('<|im_end|>') as number;
      const stopped = ids[ids.length - 1] === endId ? 'end' : stopper.interrupted ? 'user' : 'limit';
      post({ type: 'done', text: tokenizer.decode(ids, { skip_special_tokens: true }), count: ids.length, elapsed: (performance.now() - t0) / 1000, promptTokens, stopped });
    } else if (m.type === 'next') {
      if (!tokenizer || !model) throw new Error('model not loaded');
      const t0 = performance.now();
      const enc = tokenizer(m.text) as unknown as { input_ids: Tensor; attention_mask: Tensor };
      const ids = Array.from(enc.input_ids.tolist()[0] as bigint[]).map(Number);
      // Run one generation step exactly as chat does and capture the raw last-position logits.
      let captured: Tensor | null = null;
      class Capture extends LogitsProcessor { _call(_ids: unknown, logits: Tensor) { if (!captured) captured = logits; return logits; } }
      const procs = new LogitsProcessorList();
      procs.push(new Capture());
      await model.generate({ ...enc, max_new_tokens: 1, do_sample: false, logits_processor: procs } as never);
      if (!captured) throw new Error('no logits captured');
      const logits = captured as Tensor;
      const V = logits.dims[logits.dims.length - 1] as number;
      const all = toFloat32(logits);
      const row = all.subarray(all.length - V);
      let mx = -Infinity;
      for (let i = 0; i < V; i++) if (row[i] > mx) mx = row[i];
      let sum = 0;
      for (let i = 0; i < V; i++) sum += Math.exp(row[i] - mx);
      const best: { id: number; z: number }[] = [];
      for (let i = 0; i < V; i++) {
        if (best.length < m.k || row[i] > best[best.length - 1].z) {
          best.push({ id: i, z: row[i] });
          best.sort((a, b) => b.z - a.z);
          if (best.length > m.k) best.pop();
        }
      }
      post({
        type: 'next', promptIds: ids, pieces: ids.map((i) => tokenizer!.decode([i])),
        top: best.map((x) => ({ id: x.id, piece: tokenizer!.decode([x.id]), p: Math.exp(x.z - mx) / sum })), elapsed: (performance.now() - t0) / 1000,
      });
    }
  } catch (e) {
    post({ type: 'error', message: e instanceof Error ? e.message : String(e) });
  }
};
