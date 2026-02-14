use std::time::Instant;

use minisgl_cpu_core::{PrefixCacheManager, RadixCacheManager};

fn make_ids(base: i32, len: usize) -> Vec<i32> {
    let mut ids = Vec::with_capacity(len);
    ids.push(1);
    ids.push(2);
    for i in 0..(len - 2) {
        ids.push(((base + i as i32) % 1024) + 3);
    }
    ids
}

fn main() {
    let mut manager = RadixCacheManager::new();
    let corpus = 8_192usize;
    let seq_len = 16usize;

    for i in 0..corpus {
        let input = make_ids(i as i32, seq_len);
        let indices: Vec<i32> = (0..seq_len).map(|x| (i as i32) * 100 + x as i32).collect();
        manager
            .insert_prefix(&input, &indices)
            .expect("seed insert should succeed");
    }

    let queries: Vec<Vec<i32>> = (0..corpus)
        .map(|i| {
            let mut q = make_ids(i as i32, seq_len);
            q.push(2048 + (i as i32 % 17));
            q
        })
        .collect();

    let warmup = 20_000usize;
    for i in 0..warmup {
        let _ = manager
            .match_prefix(&queries[i % queries.len()])
            .expect("warmup match should succeed");
    }

    let iters = 300_000usize;
    let start = Instant::now();
    let mut total_cached = 0usize;
    for i in 0..iters {
        let (handle, _) = manager
            .match_prefix(&queries[i % queries.len()])
            .expect("bench match should succeed");
        total_cached += handle.cached_len;
    }
    let elapsed = start.elapsed().as_secs_f64();
    let ops_per_sec = iters as f64 / elapsed;
    let avg_cached_len = total_cached as f64 / iters as f64;

    println!("rust_match_prefix_ops_per_sec={ops_per_sec:.2}");
    println!("rust_match_prefix_avg_cached_len={avg_cached_len:.2}");
}
