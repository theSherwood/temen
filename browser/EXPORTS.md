# Browser export ABI

**Generated — do not edit by hand.** Regenerate with `cargo run --bin genexports` (in `browser/`). Every `#[no_mangle] extern "C"` export of the `temen-browser` cdylib, by driver family, against what the page's JS actually calls by name. `tests/exports_abi.rs` pins that every name the JS touches is exported, and that this file is fresh (#1414).

**285 exports** in 37 families — 238 referenced from JS, 47 referenced by nothing, 54 behind a `cfg`.

| family | exports | referenced from JS | `cfg`-gated |
|---|---:|---:|---:|
| `(corpus runners)` | 16 | 0 | 0 |
| `abi` | 1 | 1 | 0 |
| `alloc` | 1 | 1 | 0 |
| `bash` | 10 | 9 | 0 |
| `callprof` | 4 | 4 | 4 |
| `coop` | 66 | 57 | 0 |
| `dap` | 4 | 4 | 0 |
| `dealloc` | 1 | 1 | 0 |
| `detached` | 2 | 2 | 0 |
| `durable` | 4 | 4 | 0 |
| `exit` | 1 | 1 | 0 |
| `fault` | 1 | 1 | 0 |
| `foreign` | 6 | 6 | 6 |
| `framebuffer` | 4 | 4 | 0 |
| `jspb` | 2 | 2 | 0 |
| `link` | 11 | 8 | 0 |
| `mem` | 3 | 0 | 0 |
| `module` | 3 | 0 | 0 |
| `nim` | 5 | 5 | 0 |
| `notes` | 2 | 2 | 0 |
| `onramp` | 36 | 25 | 0 |
| `par` | 17 | 17 | 14 |
| `parse` | 3 | 3 | 0 |
| `pg` | 6 | 6 | 0 |
| `prep` | 1 | 1 | 0 |
| `run` | 18 | 17 | 0 |
| `run0` | 1 | 1 | 0 |
| `selfhost` | 1 | 1 | 0 |
| `set` | 1 | 1 | 0 |
| `snapshot` | 2 | 2 | 0 |
| `status` | 1 | 1 | 0 |
| `stderr` | 2 | 2 | 0 |
| `stdout` | 2 | 2 | 0 |
| `trap` | 2 | 2 | 0 |
| `warm` | 5 | 5 | 0 |
| `wasmjit` | 10 | 10 | 0 |
| `x2` | 30 | 30 | 30 |

## Exports by family

An export marked *unreferenced* is called by no JS or HTML in `browser/`; one marked with a `cfg` exists only in builds where that cfg holds, so a JS caller on another build gets a missing function at runtime (the source-level scan cannot see which build a page runs).

### `(corpus runners)`

- `run_capture` — *unreferenced*
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
- `temen_coop_cap_len`
- `temen_coop_cap_ptr`
- `temen_coop_close`
- `temen_coop_deliver`
- `temen_coop_deliver_cap`
- `temen_coop_deliver_fault_addr`
- `temen_coop_deliver_jit`
- `temen_coop_deliver_jit_trap`
- `temen_coop_deliver_trap`
- `temen_coop_fs_image`
- `temen_coop_fs_ptr`
- `temen_coop_func`
- `temen_coop_idle_timer`
- `temen_coop_jit_code`
- `temen_coop_jit_param_types_ptr`
- `temen_coop_jit_result_types_len`
- `temen_coop_jit_result_types_ptr`
- `temen_coop_jit_wasm_by_handle_len` — *unreferenced*
- `temen_coop_jit_wasm_by_handle_ptr`
- `temen_coop_jit_wasm_by_slot_len`
- `temen_coop_jit_wasm_len`
- `temen_coop_jit_wasm_ptr`
- `temen_coop_leaf_emits`
- `temen_coop_mapped`
- `temen_coop_mapped_now`
- `temen_coop_module`
- `temen_coop_nfuncs`
- `temen_coop_open`
- `temen_coop_open_fs`
- `temen_coop_open_selfhost_fs`
- `temen_coop_paged`
- `temen_coop_pagestate_len` — *unreferenced*
- `temen_coop_pagestate_ptr`
- `temen_coop_pass_time`
- `temen_coop_program_key`
- `temen_coop_program_key_ptr`
- `temen_coop_program_leaf`
- `temen_coop_program_wasm_len`
- `temen_coop_program_wasm_ptr`
- `temen_coop_read` — *unreferenced*
- `temen_coop_read_ptr` — *unreferenced*
- `temen_coop_region_emits` — *unreferenced*
- `temen_coop_reset_hotness`
- `temen_coop_run`
- `temen_coop_run_for`
- `temen_coop_set_emit_cap` — *unreferenced*
- `temen_coop_set_emitted_work`
- `temen_coop_set_hotness`
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
- `temen_detached_pagestate_off`

### `durable`

- `temen_durable_art_len`
- `temen_durable_art_ptr`
- `temen_durable_freeze`
- `temen_durable_thaw_resume`

### `exit`

- `temen_exit_code`

### `fault`

- `temen_fault_addr`

### `foreign`

- `temen_foreign_bench` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_foreign_poke` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_foreign_read_words` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_foreign_selftest` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_foreign_slots` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_foreign_write_words` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`

### `framebuffer`

- `temen_framebuffer_height`
- `temen_framebuffer_len`
- `temen_framebuffer_ptr`
- `temen_framebuffer_width`

### `jspb`

- `temen_jspb_read`
- `temen_jspb_write`

### `link`

- `temen_link_child_libs`
- `temen_link_encode_lib` — *unreferenced*
- `temen_link_encode_libs`
- `temen_link_lib_close`
- `temen_link_lib_open`
- `temen_link_run`
- `temen_link_run_lib`
- `temen_link_run_libs`
- `temen_link_text` — *unreferenced*
- `temen_link_text_lib` — *unreferenced*
- `temen_link_text_libs`

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
- `temen_nim_remove`

### `notes`

- `temen_notes_len`
- `temen_notes_ptr`

### `onramp`

- `temen_onramp_artifact_len` — *unreferenced*
- `temen_onramp_artifact_ptr`
- `temen_onramp_close`
- `temen_onramp_frame_begin`
- `temen_onramp_freeze`
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
- `temen_onramp_stage_budget` — *unreferenced*
- `temen_onramp_stage_clear`
- `temen_onramp_stage_js_cap`
- `temen_onramp_stage_module` — *unreferenced*
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

### `par`

- `temen_par_alloc`
- `temen_par_jit_codegen_service` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_par_jit_set_codegen` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_par_last_panic_len`
- `temen_par_last_panic_ptr`
- `temen_par_powerbox` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_par_powerbox_inst` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_par_powerbox_io` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_par_powerbox_jit_codegen` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_par_powerbox_jit_runtime` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_par_powerbox_none` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_par_powerbox_onramp` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_par_set_workers` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_par_stderr_len` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_par_stderr_ptr` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_par_stdout_len` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_par_stdout_ptr` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`

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

### `run`

- `temen_run`
- `temen_run_bash`
- `temen_run_bench`
- `temen_run_capture`
- `temen_run_durable`
- `temen_run_dynlink`
- `temen_run_jit`
- `temen_run_nested`
- `temen_run_onramp`
- `temen_run_onramp_fs`
- `temen_run_onramp_posix` — *unreferenced*
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

- `temen_trap_len`
- `temen_trap_ptr`

### `warm`

- `temen_warm_close`
- `temen_warm_coop_open`
- `temen_warm_coop_prepare`
- `temen_warm_eval`
- `temen_warm_open`

### `wasmjit`

- `temen_wasmjit_call_interp`
- `temen_wasmjit_compile`
- `temen_wasmjit_compile_b2`
- `temen_wasmjit_compile_full`
- `temen_wasmjit_env_bytes`
- `temen_wasmjit_fault_off`
- `temen_wasmjit_init_window`
- `temen_wasmjit_len`
- `temen_wasmjit_ptr`
- `temen_wasmjit_spill_sp_off`

### `x2`

- `temen_x2_answer` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_x2_bytes_ptr` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_x2_call_interp` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_x2_deliver_fault_addr` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_x2_feed` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_x2_feed_cap` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_x2_feed_end` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_x2_feed_ptr` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_x2_leaf_len` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_x2_mapped_now` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_x2_nfuncs` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_x2_open` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_x2_paged` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_x2_pagestate_len` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_x2_pagestate_ptr` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_x2_program_len` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_x2_run` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_x2_shim_len` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_x2_slot_unit` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_x2_slot_wasm_len` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_x2_spill_bytes` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_x2_spill_ptr` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_x2_stop` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_x2_table_gen` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_x2_table_log2` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_x2_thread` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_x2_wait` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_x2_wake` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_x2_wasm_len` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`
- `temen_x2_wasm_ptr` — `cfg(all(target_arch = "wasm32", target_feature = "atomics"))`

