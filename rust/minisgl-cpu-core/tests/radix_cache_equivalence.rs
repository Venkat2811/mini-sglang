use minisgl_cpu_core::{PrefixCacheManager, RadixCacheManager};

#[test]
fn exact_prefix_match_returns_expected_indices() {
    let mut mgr = RadixCacheManager::new();
    let inserted = mgr
        .insert_prefix(&[1, 2, 3], &[10, 11, 12])
        .expect("insert must succeed");
    assert_eq!(inserted, 0);

    let (handle, matched) = mgr
        .match_prefix(&[1, 2, 3, 4])
        .expect("match must succeed");
    assert_eq!(handle.cached_len, 3);
    assert_eq!(matched, vec![10, 11, 12]);
    mgr.check_integrity().expect("tree must stay valid");
}

#[test]
fn partial_match_splits_node_and_preserves_shared_prefix() {
    let mut mgr = RadixCacheManager::new();
    mgr.insert_prefix(&[1, 2, 3, 4], &[10, 11, 12, 13])
        .expect("insert seed branch");

    let (handle, matched) = mgr.match_prefix(&[1, 2, 9]).expect("partial match");
    assert_eq!(handle.cached_len, 2);
    assert_eq!(matched, vec![10, 11]);

    let prefix_len = mgr
        .insert_prefix(&[1, 2, 9], &[20, 21, 22])
        .expect("insert split branch");
    assert_eq!(prefix_len, 2);

    let (branch_handle, branch_match) = mgr
        .match_prefix(&[1, 2, 9, 8])
        .expect("match new branch");
    assert_eq!(branch_handle.cached_len, 3);
    assert_eq!(branch_match, vec![10, 11, 22]);
    mgr.check_integrity().expect("tree must stay valid");
}

#[test]
fn lock_unlock_updates_protected_and_evictable_sizes() {
    let mut mgr = RadixCacheManager::new();
    mgr.insert_prefix(&[1, 2, 3], &[7, 8, 9])
        .expect("seed insert");

    let (handle, _) = mgr.match_prefix(&[1, 2, 3]).expect("match seed");
    assert_eq!(mgr.size_info().evictable_size, 3);
    assert_eq!(mgr.size_info().protected_size, 0);

    mgr.lock_handle(&handle, false).expect("first lock");
    assert_eq!(mgr.size_info().evictable_size, 0);
    assert_eq!(mgr.size_info().protected_size, 3);

    mgr.lock_handle(&handle, false).expect("nested lock");
    assert_eq!(mgr.size_info().evictable_size, 0);
    assert_eq!(mgr.size_info().protected_size, 3);

    mgr.lock_handle(&handle, true).expect("nested unlock");
    assert_eq!(mgr.size_info().evictable_size, 0);
    assert_eq!(mgr.size_info().protected_size, 3);

    mgr.lock_handle(&handle, true).expect("final unlock");
    assert_eq!(mgr.size_info().evictable_size, 3);
    assert_eq!(mgr.size_info().protected_size, 0);
    mgr.check_integrity().expect("tree must stay valid");
}

#[test]
fn eviction_prefers_leaves_and_retains_shared_parent_prefix() {
    let mut mgr = RadixCacheManager::new();
    mgr.insert_prefix(&[1, 2, 3], &[30, 31, 32])
        .expect("insert branch a");
    mgr.insert_prefix(&[1, 2, 4], &[30, 31, 42])
        .expect("insert branch b");

    assert_eq!(mgr.size_info().evictable_size, 4);
    let evicted = mgr.evict(2).expect("evict leaves");
    assert_eq!(evicted.len(), 2);
    assert!(evicted.contains(&32));
    assert!(evicted.contains(&42));

    let (handle, matched) = mgr
        .match_prefix(&[1, 2, 3, 5])
        .expect("shared prefix survives");
    assert_eq!(handle.cached_len, 2);
    assert_eq!(matched, vec![30, 31]);
    assert_eq!(mgr.size_info().evictable_size, 2);
    mgr.check_integrity().expect("tree must stay valid");
}

#[test]
fn size_accounting_stays_consistent_across_operation_sequence() {
    let mut mgr = RadixCacheManager::new();
    let inserts = [
        vec![5, 1, 2, 3],
        vec![5, 1, 2, 4],
        vec![5, 9, 8],
        vec![7, 7, 7, 1],
        vec![7, 7, 8],
    ];

    for (idx, ids) in inserts.iter().enumerate() {
        let values: Vec<i32> = (0..ids.len())
            .map(|offset| (idx as i32) * 100 + offset as i32)
            .collect();
        mgr.insert_prefix(ids, &values).expect("insert");
        mgr.check_integrity().expect("post-insert integrity");
    }

    let mut locked = Vec::new();
    for query in [&[5, 1, 2, 3, 6][..], &[5, 9, 8, 2][..], &[7, 7, 7, 1, 0][..]] {
        let (handle, _) = mgr.match_prefix(query).expect("match");
        mgr.lock_handle(&handle, false).expect("lock");
        locked.push(handle);
        mgr.check_integrity().expect("post-lock integrity");
    }

    for handle in locked.iter().rev() {
        mgr.lock_handle(handle, true).expect("unlock");
        mgr.check_integrity().expect("post-unlock integrity");
    }

    while mgr.size_info().evictable_size > 0 {
        mgr.evict(1).expect("evict one unit");
        mgr.check_integrity().expect("post-evict integrity");
    }

    assert_eq!(mgr.size_info().evictable_size, 0);
    assert_eq!(mgr.size_info().protected_size, 0);
}
