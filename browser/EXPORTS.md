# Browser export ABI

**Generated — do not edit by hand.** Regenerate with `cargo run --bin genexports` (in `browser/`). Every `#[no_mangle] extern "C"` export of the `temen-browser` cdylib, by driver family, against what the page's JS actually calls by name. `tests/exports_abi.rs` pins that every name the JS touches is exported, and that this file is fresh (#1414).

**353 exports** in 37 families — 298 referenced from JS, 55 referenced by nothing, 10 behind a `cfg`.

| family | exports | referenced from JS | `cfg`-gated |
|---|---:|---:|---:|
| `(corpus runners)` | 17 | 0 | 0 |
| `abi` | 1 | 1 | 0 |
| `alloc` | 1 | 1 | 0 |
| `bash` | 10 | 9 | 0 |
| `callprof` | 4 | 4 | 4 |
| `coop` | 46 | 40 | 0 |
| `dap` | 4 | 4 | 0 |
| `dealloc` | 1 | 1 | 0 |
| `detached` | 5 | 5 | 1 |
| `durable` | 4 | 4 | 0 |
| `exit` | 1 | 1 | 0 |
| `fault` | 1 | 0 | 0 |
| `foreign` | 3 | 3 | 3 |
| `framebuffer` | 4 | 4 | 0 |
| `jspb` | 7 | 7 | 0 |
| `link` | 10 | 6 | 0 |
| `mem` | 3 | 0 | 0 |
| `module` | 3 | 0 | 0 |
| `nim` | 4 | 4 | 0 |
| `onramp` | 66 | 55 | 0 |
| `op13jit` | 9 | 9 | 0 |
| `par` | 74 | 74 | 2 |
| `parse` | 3 | 3 | 0 |
| `pg` | 6 | 6 | 0 |
| `prep` | 1 | 1 | 0 |
| `release` | 6 | 0 | 0 |
| `run` | 21 | 20 | 0 |
| `run0` | 1 | 1 | 0 |
| `selfhost` | 2 | 2 | 0 |
| `set` | 1 | 1 | 0 |
| `snapshot` | 2 | 2 | 0 |
| `status` | 1 | 1 | 0 |
| `stderr` | 2 | 2 | 0 |
| `stdout` | 2 | 2 | 0 |
| `trap` | 2 | 0 | 0 |
| `warm` | 16 | 16 | 0 |
| `wasmjit` | 9 | 9 | 0 |

## Exports by family

An export marked *unreferenced* is called by no JS or HTML in `browser/`; one marked with a `cfg` exists only in builds where that cfg holds, so a JS caller on another build gets a missing function at runtime (the source-level scan cannot see which build a page runs).

### `(corpus runners)`

- `run_capture` — *unreferenced*
- `run_coroutine` — *unreferenced*
- `run_durable` — *unreferenced*
- `run_dynlink` — *unreferenced*
- `run_fiber` — *unreferenced*
- `run_float` — *unreferenced*
- `run_fork` — *unreferenced*
- `run_gcroots` — *unreferenced*
- `run_guest` — *unreferenced*
- `run_jit` — *unreferenced*
- `run_powerbox` — *unreferenced*
- `run_reflect` — *unreferenced*
- `run_region` — *unreferenced*
- `run_roundtrip` — *unreferenced*
- `run_simd` — *unreferenced*
- `run_tailcall` — *unreferenced*
- `run_threads` — *unreferenced*

### `abi`

- `temen_abi_is64`

### `alloc`

- `temen_alloc`

### `bash`

- `temen_bash_coop_close` — *unreferenced*
- `temen_bash_coop_drain`
- `temen_bash_coop_exit`
- `temen_bash_coop_feed`
- `temen_bash_coop_open`
- `temen_bash_coop_pump`
- `temen_bash_drain`
- `temen_bash_exited`
- `temen_bash_feed`
- `temen_bash_session`

### `callprof`

- `temen_callprof_dump` — `cfg(feature = "callprof")`
- `temen_callprof_len` — `cfg(feature = "callprof")`
- `temen_callprof_ptr` — `cfg(feature = "callprof")`
- `temen_callprof_reset` — `cfg(feature = "callprof")`

### `coop`

- `temen_coop_argv_len`
- `temen_coop_argv_ptr`
- `temen_coop_call_interp`
- `temen_coop_close`
- `temen_coop_deliver`
- `temen_coop_deliver_jit`
- `temen_coop_deliver_jit_trap`
- `temen_coop_deliver_trap`
- `temen_coop_func`
- `temen_coop_jit_code`
- `temen_coop_jit_param_types_ptr`
- `temen_coop_jit_result_types_len`
- `temen_coop_jit_result_types_ptr`
- `temen_coop_jit_wasm_by_handle_len` — *unreferenced*
- `temen_coop_jit_wasm_by_handle_ptr`
- `temen_coop_jit_wasm_by_slot_len`
- `temen_coop_jit_wasm_len`
- `temen_coop_jit_wasm_ptr`
- `temen_coop_leaf_wasm_len`
- `temen_coop_leaf_wasm_ptr`
- `temen_coop_mapped`
- `temen_coop_mapped_now`
- `temen_coop_module`
- `temen_coop_nfuncs`
- `temen_coop_open`
- `temen_coop_paged`
- `temen_coop_pagestate_len` — *unreferenced*
- `temen_coop_pagestate_ptr`
- `temen_coop_run`
- `temen_coop_set_emit_cap` — *unreferenced*
- `temen_coop_set_tierup_floor`
- `temen_coop_shim_ptr`
- `temen_coop_shim_wasm`
- `temen_coop_slot_unit`
- `temen_coop_spill_bytes`
- `temen_coop_spill_ptr`
- `temen_coop_table_gen`
- `temen_coop_table_log2`
- `temen_coop_task`
- `temen_coop_tierup_win_len` — *unreferenced*
- `temen_coop_tierup_win_ptr`
- `temen_coop_value` — *unreferenced*
- `temen_coop_wasm_len`
- `temen_coop_wasm_ptr`
- `temen_coop_win_len`
- `temen_coop_win_ptr` — *unreferenced*

### `dap`

- `temen_dap_request`
- `temen_dap_reset`
- `temen_dap_response_len`
- `temen_dap_response_ptr`

### `dealloc`

- `temen_dealloc`

### `detached`

- `temen_detached_header_bytes`
- `temen_detached_jit_run_open` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_detached_max_bytes`
- `temen_detached_oracle_run`
- `temen_detached_pagestate_off`

### `durable`

- `temen_durable_art_len`
- `temen_durable_art_ptr`
- `temen_durable_freeze`
- `temen_durable_thaw_resume`

### `exit`

- `temen_exit_code`

### `fault`

- `temen_fault_addr` — *unreferenced*

### `foreign`

- `temen_foreign_bench` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_foreign_poke` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_foreign_selftest` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`

### `framebuffer`

- `temen_framebuffer_height`
- `temen_framebuffer_len`
- `temen_framebuffer_ptr`
- `temen_framebuffer_width`

### `jspb`

- `temen_jspb_bind`
- `temen_jspb_error_len`
- `temen_jspb_error_ptr`
- `temen_jspb_read`
- `temen_jspb_reset`
- `temen_jspb_run`
- `temen_jspb_write`

### `link`

- `temen_link_encode_lib`
- `temen_link_encode_libs` — *unreferenced*
- `temen_link_lib_close`
- `temen_link_lib_open`
- `temen_link_run`
- `temen_link_run_lib`
- `temen_link_run_libs` — *unreferenced*
- `temen_link_text` — *unreferenced*
- `temen_link_text_lib`
- `temen_link_text_libs` — *unreferenced*

### `mem`

- `temen_mem_profile` — *unreferenced*
- `temen_mem_profile_stats_len` — *unreferenced*
- `temen_mem_profile_stats_ptr` — *unreferenced*

### `module`

- `temen_module_imports` — *unreferenced*
- `temen_module_imports_len` — *unreferenced*
- `temen_module_imports_ptr` — *unreferenced*

### `nim`

- `temen_nim_file`
- `temen_nim_file_ptr`
- `temen_nim_module_suffix`
- `temen_nim_open`

### `onramp`

- `temen_onramp_artifact_len` — *unreferenced*
- `temen_onramp_artifact_ptr`
- `temen_onramp_close`
- `temen_onramp_frame`
- `temen_onramp_freeze`
- `temen_onramp_jit_call_interp`
- `temen_onramp_jit_close`
- `temen_onramp_jit_entry_sp`
- `temen_onramp_jit_env_bytes`
- `temen_onramp_jit_freeze`
- `temen_onramp_jit_key`
- `temen_onramp_jit_moment_restore` — *unreferenced*
- `temen_onramp_jit_moment_take` — *unreferenced*
- `temen_onramp_jit_mouse`
- `temen_onramp_jit_open`
- `temen_onramp_jit_open_fs`
- `temen_onramp_jit_present`
- `temen_onramp_jit_run_call_interp`
- `temen_onramp_jit_run_close`
- `temen_onramp_jit_run_env_bytes`
- `temen_onramp_jit_run_finish`
- `temen_onramp_jit_run_mapped`
- `temen_onramp_jit_run_open`
- `temen_onramp_jit_run_open_fs`
- `temen_onramp_jit_run_pagestate_len`
- `temen_onramp_jit_run_pagestate_ptr`
- `temen_onramp_jit_run_report`
- `temen_onramp_jit_run_slot`
- `temen_onramp_jit_run_slot_count`
- `temen_onramp_jit_run_trap_len`
- `temen_onramp_jit_run_wasm_len`
- `temen_onramp_jit_run_wasm_ptr`
- `temen_onramp_jit_run_win_ptr`
- `temen_onramp_jit_thaw`
- `temen_onramp_jit_tick`
- `temen_onramp_jit_trap_len`
- `temen_onramp_jit_wasm_len`
- `temen_onramp_jit_wasm_ptr`
- `temen_onramp_jit_win_ptr`
- `temen_onramp_key`
- `temen_onramp_moment_bytes` — *unreferenced*
- `temen_onramp_moment_clear` — *unreferenced*
- `temen_onramp_moment_free` — *unreferenced*
- `temen_onramp_moment_restore` — *unreferenced*
- `temen_onramp_moment_take` — *unreferenced*
- `temen_onramp_mouse`
- `temen_onramp_open`
- `temen_onramp_open_fs`
- `temen_onramp_set_grant_instantiator` — *unreferenced*
- `temen_onramp_thaw`
- `temen_onramp_timeline_begin_tick`
- `temen_onramp_timeline_close`
- `temen_onramp_timeline_end_tick`
- `temen_onramp_timeline_held_bytes` — *unreferenced*
- `temen_onramp_timeline_key`
- `temen_onramp_timeline_keyframe_at`
- `temen_onramp_timeline_keyframe_count`
- `temen_onramp_timeline_len`
- `temen_onramp_timeline_mouse` — *unreferenced*
- `temen_onramp_timeline_open`
- `temen_onramp_timeline_seek_begin`
- `temen_onramp_timeline_taped_at`
- `temen_onramp_timeline_taped_count`
- `temen_onramp_timeline_tick`
- `temen_onramp_timeline_truncate`
- `temen_onramp_trap_len`

### `op13jit`

- `temen_op13jit_child_mem_id`
- `temen_op13jit_close`
- `temen_op13jit_counter`
- `temen_op13jit_deliver`
- `temen_op13jit_open`
- `temen_op13jit_open_child`
- `temen_op13jit_open_detached`
- `temen_op13jit_result`
- `temen_op13jit_step`

### `par`

- `temen_par_alloc`
- `temen_par_child`
- `temen_par_child_detached` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_par_compile`
- `temen_par_compile_jit`
- `temen_par_deliver_child`
- `temen_par_deliver_code`
- `temen_par_deliver_jit_invoke`
- `temen_par_deliver_jit_invoke_trap`
- `temen_par_deliver_join`
- `temen_par_deliver_tierup`
- `temen_par_deliver_tierup_trap`
- `temen_par_enable_inst_codegen`
- `temen_par_enable_jit`
- `temen_par_enable_jit_codegen`
- `temen_par_enable_jit_paged`
- `temen_par_ev_a`
- `temen_par_ev_b`
- `temen_par_ev_c`
- `temen_par_ev_d`
- `temen_par_free`
- `temen_par_inst_call_interp`
- `temen_par_inst_eligible`
- `temen_par_inst_end_join`
- `temen_par_inst_file_child`
- `temen_par_inst_instantiate_rec`
- `temen_par_inst_join`
- `temen_par_inst_paged`
- `temen_par_inst_pagestate_sync`
- `temen_par_inst_unit_wasm_len`
- `temen_par_inst_unit_wasm_ptr`
- `temen_par_jit_argv_len`
- `temen_par_jit_argv_ptr`
- `temen_par_jit_code`
- `temen_par_jit_code_wasm_len`
- `temen_par_jit_code_wasm_ptr`
- `temen_par_jit_codegen_service`
- `temen_par_jit_param_types_ptr`
- `temen_par_jit_result_types_len`
- `temen_par_jit_result_types_ptr`
- `temen_par_jit_set_b2`
- `temen_par_jit_set_codegen`
- `temen_par_jit_slot_unit`
- `temen_par_jit_table_gen`
- `temen_par_jit_table_log2`
- `temen_par_jit_unit_wasm_by_slot_len`
- `temen_par_jit_unit_wasm_by_slot_ptr`
- `temen_par_jit_unit_wasm_len`
- `temen_par_jit_unit_wasm_ptr`
- `temen_par_last_panic_len`
- `temen_par_last_panic_ptr`
- `temen_par_nfuncs`
- `temen_par_powerbox`
- `temen_par_powerbox_inst`
- `temen_par_powerbox_io`
- `temen_par_powerbox_jit_codegen`
- `temen_par_powerbox_jit_runtime`
- `temen_par_powerbox_none`
- `temen_par_powerbox_onramp`
- `temen_par_root`
- `temen_par_root_call_interp`
- `temen_par_run`
- `temen_par_shim_wasm_len`
- `temen_par_shim_wasm_ptr`
- `temen_par_stderr_len`
- `temen_par_stderr_ptr`
- `temen_par_stdout_len`
- `temen_par_stdout_ptr`
- `temen_par_thread_detached` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_par_tierup_argv_len`
- `temen_par_tierup_argv_ptr`
- `temen_par_tierup_pagestate_len`
- `temen_par_tierup_pagestate_ptr`
- `temen_par_win_len`

### `parse`

- `temen_parse`
- `temen_parse_len`
- `temen_parse_ptr`

### `pg`

- `temen_pg_close`
- `temen_pg_open`
- `temen_pg_query`
- `temen_pg_snapshot`
- `temen_pg_snapshot_len`
- `temen_pg_snapshot_ptr`

### `prep`

- `temen_prep_bench`

### `release`

- `temen_release_close` — *unreferenced*
- `temen_release_fs_image` — *unreferenced*
- `temen_release_fs_ptr` — *unreferenced*
- `temen_release_open` — *unreferenced*
- `temen_release_run` — *unreferenced*
- `temen_release_value` — *unreferenced*

### `run`

- `temen_run`
- `temen_run_bash`
- `temen_run_bench`
- `temen_run_capture`
- `temen_run_durable`
- `temen_run_dynlink`
- `temen_run_jit`
- `temen_run_nested`
- `temen_run_nifler_fs`
- `temen_run_nifler_jit_open`
- `temen_run_onramp`
- `temen_run_onramp_fs`
- `temen_run_onramp_posix` — *unreferenced*
- `temen_run_onramp_stream`
- `temen_run_pb`
- `temen_run_pg`
- `temen_run_reflect`
- `temen_run_region`
- `temen_run_shared`
- `temen_run_shell`
- `temen_run_value`

### `run0`

- `temen_run0`

### `selfhost`

- `temen_selfhost_emit_object_fs`
- `temen_selfhost_jit_emit_object_fs`

### `set`

- `temen_set_run_env`

### `snapshot`

- `temen_snapshot_len`
- `temen_snapshot_ptr`

### `status`

- `temen_status`

### `stderr`

- `temen_stderr_len`
- `temen_stderr_ptr`

### `stdout`

- `temen_stdout_len`
- `temen_stdout_ptr`

### `trap`

- `temen_trap_len` — *unreferenced*
- `temen_trap_ptr` — *unreferenced*

### `warm`

- `temen_warm_close`
- `temen_warm_coop_open`
- `temen_warm_coop_prepare`
- `temen_warm_eval`
- `temen_warm_jit_call_interp`
- `temen_warm_jit_entry_func`
- `temen_warm_jit_entry_sp`
- `temen_warm_jit_finish`
- `temen_warm_jit_open`
- `temen_warm_jit_prepare`
- `temen_warm_jit_report`
- `temen_warm_jit_set_split`
- `temen_warm_jit_wasm_len`
- `temen_warm_jit_wasm_ptr`
- `temen_warm_jit_win_ptr`
- `temen_warm_open`

### `wasmjit`

- `temen_wasmjit_call_interp`
- `temen_wasmjit_compile`
- `temen_wasmjit_compile_b2`
- `temen_wasmjit_compile_full`
- `temen_wasmjit_env_bytes`
- `temen_wasmjit_init_window`
- `temen_wasmjit_len`
- `temen_wasmjit_ptr`
- `temen_wasmjit_spill_sp_off`

