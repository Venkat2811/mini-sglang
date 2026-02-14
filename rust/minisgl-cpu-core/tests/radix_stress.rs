use minisgl_cpu_core::{PrefixCacheManager, RadixCacheHandle, RadixCacheManager};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn random_seq(rng: &mut StdRng, min_len: usize, max_len: usize) -> Vec<i32> {
    let len = rng.random_range(min_len..=max_len);
    (0..len).map(|_| rng.random_range(1..=512)).collect()
}

#[test]
fn randomized_operations_preserve_integrity() {
    let mut rng = StdRng::seed_from_u64(0xCAFE_BABE);
    let mut manager = RadixCacheManager::new();
    let mut inserted: Vec<Vec<i32>> = Vec::new();
    let mut locked_handles: Vec<RadixCacheHandle> = Vec::new();

    for step in 0_i32..1_000_i32 {
        let action = rng.random_range(0..5);
        match action {
            0 | 1 => {
                let ids = random_seq(&mut rng, 2, 12);
                let indices: Vec<i32> = (0..ids.len())
                    .map(|idx| step * 1_000 + idx as i32)
                    .collect();
                manager
                    .insert_prefix(&ids, &indices)
                    .expect("random insert should succeed");
                inserted.push(ids);
            }
            2 => {
                let input = if inserted.is_empty() || rng.random_bool(0.35) {
                    random_seq(&mut rng, 2, 12)
                } else {
                    let mut picked = inserted[rng.random_range(0..inserted.len())].clone();
                    if rng.random_bool(0.5) {
                        picked.push(rng.random_range(1..=512));
                    }
                    picked
                };
                let (handle, _) = manager
                    .match_prefix(&input)
                    .expect("random match should succeed");
                if handle.cached_len > 0 && rng.random_bool(0.35) {
                    manager
                        .lock_handle(&handle, false)
                        .expect("random lock should succeed");
                    locked_handles.push(handle);
                }
            }
            3 => {
                if let Some(handle) = locked_handles.pop() {
                    manager
                        .lock_handle(&handle, true)
                        .expect("random unlock should succeed");
                }
            }
            _ => {
                let evictable = manager.size_info().evictable_size;
                if evictable > 0 {
                    let size = rng.random_range(1..=evictable.min(3));
                    let _ = manager.evict(size).expect("random evict should succeed");
                }
            }
        }

        manager
            .check_integrity()
            .expect("integrity must hold throughout stress run");
    }

    for handle in locked_handles.iter().rev() {
        manager
            .lock_handle(handle, true)
            .expect("unlock remaining locked handle");
        manager
            .check_integrity()
            .expect("integrity must hold after final unlocks");
    }
}
