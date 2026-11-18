# Lightweight diagnostic storage

The default training objective, rollout schedule, 256-token stratified parameter-gradient sample,
diagnostic observation schedule and numerical precision are unchanged.

## Files retained

| Records | Contents |
| --- | --- |
| `training_records/rq1_step_*.json` | Existing per-update token statistics and weighted parameter-gradient sample records |
| `training_records/rq2/step_*.npz` | Existing accumulated-change spectra, bases and coverage; incremental-change summaries below |
| `training_records/rq2/gradient_*.npz` | Existing hidden-gradient spectra and early-basis coverage |
| `training_records/rq2/probe_initial.npz` | Fixed probe inputs, positions and initial hidden states |
| `training_records/rq2/reference_010.npz` | One fixed step-10 reference, for incremental measurements and resume |
| `training_records/parameters/update_*.json` | Actual parameter-update summaries at existing validation diagnostics |
| `validation128/` | Split and fresh validation responses; always inside the training output, even with external weights |
| `replication/` | Existing detailed statistics, including losslessly stored second moments |
| `eval/`, `logs/` | Existing answers, scores, manifests and logs |

Full hidden-gradient sample vectors (`token_sample_*.npz`) are no longer written by default.
All computations that produced the existing summary statistics still run. The optional
`replication17.save_hidden_gradient_samples=true` restores the original sample archives.
Per-token full-model parameter-gradient vectors are not saved; sampled squared norms,
positions, weights and estimation metadata are retained as before.

## Lossless second moments

Symmetric `second` matrices are stored as `second_upper`, `second_dimension` and
`storage_schema=symmetric-upper-v1`. This retains the exact numerical values and dtype;
the raw matrix payload contains `d(d+1)/2` values instead of `d*d`. Asymmetric matrices
fall back to the legacy dense representation. Deflate level 1 reduces compression work.
Actual compressed file sizes depend on the data, and are not guaranteed to halve.

Every result ZIP includes `READ_MOMENTS.py`. With NumPy installed, use:

```python
from READ_MOMENTS import load_moment
arrays = load_moment("path/to/moments.npz")
second = arrays["second"]
```

The reader accepts old dense archives as well. Existing consumers that directly index
`np.load(path)["second"]` must use this reader for newly packed moment files. Ordinary
RQ2 spectrum/basis files retain their original keys.

## Newly recorded measurements

For fixed probe states normalized per position, each RQ2 observation records the squared
Frobenius norm, Frobenius norm and position RMS of `h_t - h_10` and `h_t - h_(t-1)`.
It also records their energy coverage by the fixed step-10 basis when that basis and
nonzero incremental energy are available. Only existing probe forward outputs are used;
no additional model pass or eigendecomposition is performed. `since_step10_*` and
`previous_step_*` identify the two comparisons. Zero-energy coverage is left undefined.
Changes are subtracted before projection, in bounded CPU blocks, to avoid cancellation
when late-training changes are much smaller than the individual hidden states.

The previous hidden state is held in CPU memory, not written every step. After a process
restart the first adjacent-step comparison has `previous_step_available=false`; subsequent
steps are available. Step-10-relative comparisons resume from the single saved reference.
Old runs without this reference cannot retrospectively supply it.
When rolling back to a checkpoint before step 10, the observation basis is restored from
that committed history. A later basis left by the interrupted attempt is never used.

At existing validation diagnostics, parameter summaries reuse before/after buffers and
record exact nonzero-coordinate counts, positive/negative counts, L1/L2 sizes, RMS, maximum
change, relative update norm, energy shares and train/validation direction cosines, by
layer and existing parameter group. FSDP padding is excluded and tied weights count once.
These describe the actual stored parameter changes, not unrounded optimizer proposals.
Group categories overlap; `all` is the total. No new full-model snapshots, backward passes
or parameter SVDs are added. Cumulative parameter rank and cross-time parameter directions
are not claimed by these records.

Small extra CPU reductions are timed (`timing/rq2_increment_summary_seconds` and
`parameter_update_summary_seconds`). Zero wall-clock overhead is not guaranteed.
Parameter dot products share FP64 block conversions and combine their distributed
exchange. Individual dot products retain the original chunk and rank reduction order;
no tokens, layers, diagnostic steps or precision are removed. Before/after-update dot
exchanges decrease from five to two per validation diagnostic.

## Model cleanup

Successful evaluation cleans checkpoints and merged weights for both normal output and
new owned `--model-artifacts-root` directories. A per-run ownership marker and completed
evaluation are required. Unknown pre-existing external directories are not adopted or
deleted. Shared caches, diagnostics and answers are never cleanup targets. Failures and
`--keep-model-files` preserve weights. Default checkpoint frequency is unchanged.
Older external runs can resume after checkpoint validation; without ownership metadata,
automatic cleanup is skipped and result packaging continues. Early-stop evaluation is
checked against the actual weight directory, even if convenience links are absent.

Original result files remain alongside the ZIP; no automatic deletion of the only source
statistics is performed. An earlier ZIP and its replacement can coexist during atomic
packaging. Updating this code affects future records and does not regenerate missing
historical measurements.

## Resume and verification

Default final-model exports cannot resume AdamW training. To support recovery, explicitly
use `--save-freq N`; resume with the same experiment, output, GPU count and save settings.
The launcher requires the committed checkpoint pointer, all model/optimizer/extra-state
shards, data cursor and consistent rollout/update counters. Missing checkpoints fail
instead of starting a fresh model against existing statistics.

`tests/check_storage_four_rank.py` runs four real CPU/Gloo FSDP processes. It checks
shard padding, tied weights, exact equality to the legacy dot reductions, agreement
between ranks, lossless archives and propagation of writer failures to every rank.
The primary rank writes moment archives and reports success/failure before the next
collective. GPU/NCCL throughput still requires measurement on the training server.
