// Python examples shown in the course. The source files live in examples/python and are
// imported verbatim; their outputs were recorded by scripts/run_examples.py (real runs).
import ex01 from '../../examples/python/01_neuron.py?raw';
import ex02 from '../../examples/python/02_vectors.py?raw';
import ex03 from '../../examples/python/03_softmax_xent.py?raw';
import ex04 from '../../examples/python/04_tokenizer.py?raw';
import ex05 from '../../examples/python/05_embedding_rmsnorm.py?raw';
import ex06 from '../../examples/python/06_attention.py?raw';
import ex07 from '../../examples/python/07_gqa_rope.py?raw';
import ex08 from '../../examples/python/08_ffn_moe.py?raw';
import ex09 from '../../examples/python/09_tiny_transformer.py?raw';
import ex10 from '../../examples/python/10_backprop.py?raw';
import ex11 from '../../examples/python/11_train_tiny_lm.py?raw';
import ex12 from '../../examples/python/12_hf_generate.py?raw';
import ex13 from '../../examples/python/13_hf_inspect.py?raw';
import ex14 from '../../examples/python/14_safetensors.py?raw';
import ex15 from '../../examples/python/15_quantize.py?raw';
import ex16 from '../../examples/python/16_api_client.py?raw';
import outputs from './code-outputs.json';

export type CodeId =
  | 'neuron' | 'vectors' | 'softmax' | 'tokenizer' | 'embed-norm' | 'attention' | 'gqa-rope' | 'ffn-moe'
  | 'tiny-transformer' | 'backprop' | 'train-lm' | 'hf-generate' | 'hf-inspect' | 'safetensors' | 'quantize' | 'api-client';

export interface CodeExample { id: CodeId; file: string; source: string; needs: string; output: string | null; ran: boolean; note?: string; realModel: boolean }

type Rec = { ran: boolean; stdout?: string; note?: string };
const OUT = outputs as unknown as Record<string, Rec> & { _env: { python: string; numpy: string; torch: string; transformers: string; date: string; device: string } };

const def = (id: CodeId, file: string, source: string, needs: string, realModel = false): CodeExample => {
  const r = OUT[file];
  return { id, file, source, needs, realModel, ran: !!r?.ran, output: r?.ran ? r.stdout ?? '' : null, note: r?.note };
};

export const CODE: Record<CodeId, CodeExample> = {
  neuron: def('neuron', '01_neuron.py', ex01, 'numpy'),
  vectors: def('vectors', '02_vectors.py', ex02, 'numpy'),
  softmax: def('softmax', '03_softmax_xent.py', ex03, 'numpy'),
  tokenizer: def('tokenizer', '04_tokenizer.py', ex04, 'transformers', true),
  'embed-norm': def('embed-norm', '05_embedding_rmsnorm.py', ex05, 'numpy'),
  attention: def('attention', '06_attention.py', ex06, 'numpy'),
  'gqa-rope': def('gqa-rope', '07_gqa_rope.py', ex07, 'numpy'),
  'ffn-moe': def('ffn-moe', '08_ffn_moe.py', ex08, 'numpy'),
  'tiny-transformer': def('tiny-transformer', '09_tiny_transformer.py', ex09, 'numpy'),
  backprop: def('backprop', '10_backprop.py', ex10, 'numpy, torch'),
  'train-lm': def('train-lm', '11_train_tiny_lm.py', ex11, 'torch'),
  'hf-generate': def('hf-generate', '12_hf_generate.py', ex12, 'torch, transformers', true),
  'hf-inspect': def('hf-inspect', '13_hf_inspect.py', ex13, 'torch, transformers', true),
  safetensors: def('safetensors', '14_safetensors.py', ex14, 'numpy (safetensors optional)', true),
  quantize: def('quantize', '15_quantize.py', ex15, 'numpy'),
  'api-client': def('api-client', '16_api_client.py', ex16, 'requests + a running server'),
};

export const CODE_ENV = OUT._env;
export const REPO_URL = 'https://github.com/sevcikro/qwen-architecture-explorer';
