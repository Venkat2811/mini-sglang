use minisgl_cpu_core::types::BatchPhase;
use minisgl_cpu_core::{Batch, CacheManager, NoopCacheManager, Req, SamplingParams, SchedulerPlan};

#[test]
fn sampling_params_defaults_match_python_contract() {
    let params = SamplingParams::default();
    assert_eq!(params.temperature, 0.0);
    assert_eq!(params.top_k, -1);
    assert_eq!(params.top_p, 1.0);
    assert!(!params.ignore_eos);
    assert_eq!(params.max_tokens, 1024);
}

#[test]
fn noop_cache_manager_handles_non_empty_input() {
    let req = Req::new(1, vec![10, 20, 30], 0, 64, SamplingParams::default());
    let mgr = NoopCacheManager;
    let matched = mgr.match_prefix(&req).expect("prefix match should work");
    assert_eq!(matched.cached_len, 0);
    assert!(matched.indices.is_empty());
}

#[test]
fn scheduler_plan_tracks_batch_uids() {
    let req_a = Req::new(100, vec![1, 2], 0, 5, SamplingParams::default());
    let req_b = Req::new(101, vec![3, 4], 0, 7, SamplingParams::default());
    let batch = Batch::new(vec![req_a, req_b], BatchPhase::Prefill);
    let plan = SchedulerPlan::from_batch(&batch);
    assert_eq!(plan.selected_uids, vec![100, 101]);
    assert_eq!(plan.phase, BatchPhase::Prefill);
}
