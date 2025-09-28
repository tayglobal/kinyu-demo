/* tslint:disable */
/* eslint-disable */
export function price_exotic_warrant_wasm(s0: number, strike_discount: number, buyback_price: number, t: number, forward_curve: Float64Array, sigma: number, credit_spreads: Float64Array, equity_credit_corr: number, recovery_rate: number, n_paths: number, n_steps: number, poly_degree: number, seed?: bigint | null): number;
export function test_wasm(): number;
export function simple_test_calculation(): number;
export function debug_calculation(): number;
export function simple_option_test(): number;
export function minimal_test(): number;
export function test_random_generation(): number;
export function test_random_seed_variation(): number;
export function test_price_variation_with_seeds(): number;
export function test_seed_functionality(): number;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly price_exotic_warrant_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: bigint) => number;
  readonly test_wasm: () => number;
  readonly simple_test_calculation: () => number;
  readonly debug_calculation: () => number;
  readonly simple_option_test: () => number;
  readonly minimal_test: () => number;
  readonly test_random_generation: () => number;
  readonly test_random_seed_variation: () => number;
  readonly test_price_variation_with_seeds: () => number;
  readonly test_seed_functionality: () => number;
  readonly __wbindgen_exn_store: (a: number) => void;
  readonly __externref_table_alloc: () => number;
  readonly __wbindgen_export_2: WebAssembly.Table;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
