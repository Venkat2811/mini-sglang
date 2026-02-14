use std::{
    cell::RefCell,
    cmp::{Ordering, Reverse},
    collections::{BinaryHeap, HashMap},
    rc::{Rc, Weak},
    time::{SystemTime, UNIX_EPOCH},
};

use crate::cache::{CacheError, PrefixCacheManager, SizeInfo};

type NodeRef = Rc<RefCell<RadixNode>>;

#[derive(Debug)]
struct RadixNode {
    id: u64,
    key: Vec<i32>,
    value: Vec<i32>,
    children: HashMap<i32, NodeRef>,
    parent: Option<Weak<RefCell<RadixNode>>>,
    ref_count: usize,
    timestamp: u128,
}

impl RadixNode {
    fn new(id: u64, timestamp: u128) -> Self {
        Self {
            id,
            key: Vec::new(),
            value: Vec::new(),
            children: HashMap::new(),
            parent: None,
            ref_count: 0,
            timestamp,
        }
    }

    fn len(&self) -> usize {
        self.key.len()
    }

    fn is_root(&self) -> bool {
        self.parent.is_none()
    }

    fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

#[derive(Clone, Debug)]
pub struct RadixCacheHandle {
    pub cached_len: usize,
    node: NodeRef,
}

impl RadixCacheHandle {
    fn new(cached_len: usize, node: NodeRef) -> Self {
        Self { cached_len, node }
    }
}

#[derive(Debug)]
pub struct RadixCacheManager {
    root_node: NodeRef,
    next_node_id: u64,
    evictable_size: usize,
    protected_size: usize,
}

impl Default for RadixCacheManager {
    fn default() -> Self {
        Self::new()
    }
}

impl RadixCacheManager {
    pub fn new() -> Self {
        let root = Rc::new(RefCell::new(RadixNode::new(0, Self::now_tick())));
        root.borrow_mut().ref_count = 1; // Root is always protected.
        Self {
            root_node: root,
            next_node_id: 1,
            evictable_size: 0,
            protected_size: 0,
        }
    }

    fn now_tick() -> u128 {
        match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(dur) => dur.as_nanos(),
            Err(_) => 0,
        }
    }

    fn alloc_node_id(&mut self) -> u64 {
        let id = self.next_node_id;
        self.next_node_id += 1;
        id
    }

    fn parent_of(node: &NodeRef) -> Result<NodeRef, CacheError> {
        node.borrow()
            .parent
            .as_ref()
            .and_then(Weak::upgrade)
            .ok_or(CacheError::CorruptedTree {
                reason: "missing parent pointer",
            })
    }

    fn common_prefix_len(a: &[i32], b: &[i32]) -> usize {
        a.iter()
            .zip(b.iter())
            .take_while(|(lhs, rhs)| lhs == rhs)
            .count()
    }

    fn split_node(&mut self, node: &NodeRef, pos: usize) -> Result<NodeRef, CacheError> {
        let (orig_key, orig_value, orig_ref_count, orig_timestamp) = {
            let borrowed = node.borrow();
            if pos == 0 || pos >= borrowed.key.len() {
                return Err(CacheError::CorruptedTree {
                    reason: "invalid split position",
                });
            }
            (
                borrowed.key.clone(),
                borrowed.value.clone(),
                borrowed.ref_count,
                borrowed.timestamp,
            )
        };
        let parent = Self::parent_of(node)?;

        let mut split = RadixNode::new(self.alloc_node_id(), orig_timestamp);
        split.key = orig_key[..pos].to_vec();
        split.value = orig_value[..pos].to_vec();
        split.ref_count = orig_ref_count;
        split.parent = Some(Rc::downgrade(&parent));
        let split_ref = Rc::new(RefCell::new(split));

        {
            let mut parent_mut = parent.borrow_mut();
            parent_mut.children.insert(orig_key[0], split_ref.clone());
        }

        {
            let mut node_mut = node.borrow_mut();
            node_mut.key = orig_key[pos..].to_vec();
            node_mut.value = orig_value[pos..].to_vec();
            node_mut.parent = Some(Rc::downgrade(&split_ref));
        }

        let child_edge = node.borrow().key.first().copied().ok_or(CacheError::CorruptedTree {
            reason: "split child became empty",
        })?;
        split_ref.borrow_mut().children.insert(child_edge, node.clone());

        Ok(split_ref)
    }

    fn walk(&mut self, input_ids: &[i32]) -> Result<(NodeRef, usize), CacheError> {
        let mut prefix_len = 0usize;
        let input_len = input_ids.len();
        let mut node = self.root_node.clone();
        let tick = Self::now_tick();

        while prefix_len < input_len {
            let id = input_ids[prefix_len];
            let child = {
                let borrowed = node.borrow();
                borrowed.children.get(&id).cloned()
            };
            let Some(child) = child else {
                return Ok((node, prefix_len));
            };

            let (match_len, child_len) = {
                let child_borrow = child.borrow();
                (
                    Self::common_prefix_len(&child_borrow.key, &input_ids[prefix_len..]),
                    child_borrow.len(),
                )
            };
            prefix_len += match_len;

            if match_len != child_len {
                return self
                    .split_node(&child, match_len)
                    .map(|split| (split, prefix_len));
            }

            child.borrow_mut().timestamp = tick;
            node = child;
        }

        Ok((node, prefix_len))
    }

    fn collect_leaf_nodes_for_evict(&self) -> Vec<NodeRef> {
        let mut stack = vec![self.root_node.clone()];
        let mut leaves = Vec::new();

        while let Some(node) = stack.pop() {
            let borrowed = node.borrow();
            if borrowed.is_leaf() {
                if borrowed.ref_count == 0 {
                    leaves.push(node.clone());
                }
                continue;
            }
            for child in borrowed.children.values() {
                stack.push(child.clone());
            }
        }

        leaves
    }
}

#[derive(Clone)]
struct HeapEntry {
    timestamp: u128,
    node_id: u64,
    node: NodeRef,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.timestamp == other.timestamp && self.node_id == other.node_id
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.timestamp
            .cmp(&other.timestamp)
            .then(self.node_id.cmp(&other.node_id))
    }
}

impl PrefixCacheManager for RadixCacheManager {
    type Handle = RadixCacheHandle;

    fn match_prefix(&mut self, input_ids: &[i32]) -> Result<(Self::Handle, Vec<i32>), CacheError> {
        let (node, prefix_len) = self.walk(input_ids)?;
        if prefix_len == 0 {
            return Ok((RadixCacheHandle::new(0, node), Vec::new()));
        }

        let matched_node = node.clone();
        let mut segments = Vec::<Vec<i32>>::new();
        let mut cursor = node;

        loop {
            let parent = {
                let borrowed = cursor.borrow();
                if borrowed.is_root() {
                    break;
                }
                segments.push(borrowed.value.clone());
                borrowed.parent.as_ref().and_then(Weak::upgrade)
            };
            cursor = parent.ok_or(CacheError::CorruptedTree {
                reason: "missing parent while reconstructing match",
            })?;
        }

        segments.reverse();
        let total = segments.iter().map(Vec::len).sum();
        let mut indices = Vec::with_capacity(total);
        for seg in segments {
            indices.extend(seg);
        }

        Ok((RadixCacheHandle::new(prefix_len, matched_node), indices))
    }

    fn lock_handle(&mut self, handle: &Self::Handle, unlock: bool) -> Result<(), CacheError> {
        let mut node = handle.node.clone();
        while !node.borrow().is_root() {
            if unlock {
                let mut borrowed = node.borrow_mut();
                if borrowed.ref_count == 0 {
                    return Err(CacheError::UnlockUnderflow);
                }
                borrowed.ref_count -= 1;
                if borrowed.ref_count == 0 {
                    self.evictable_size += borrowed.len();
                    self.protected_size = self
                        .protected_size
                        .checked_sub(borrowed.len())
                        .ok_or(CacheError::CorruptedTree {
                            reason: "protected_size underflow during unlock",
                        })?;
                }
            } else {
                let mut borrowed = node.borrow_mut();
                if borrowed.ref_count == 0 {
                    self.evictable_size = self.evictable_size.checked_sub(borrowed.len()).ok_or(
                        CacheError::CorruptedTree {
                            reason: "evictable_size underflow during lock",
                        },
                    )?;
                    self.protected_size += borrowed.len();
                }
                borrowed.ref_count += 1;
            }
            node = Self::parent_of(&node)?;
        }
        Ok(())
    }

    fn insert_prefix(&mut self, input_ids: &[i32], indices: &[i32]) -> Result<usize, CacheError> {
        if input_ids.len() != indices.len() {
            return Err(CacheError::MismatchedInputAndIndices {
                input_len: input_ids.len(),
                indices_len: indices.len(),
            });
        }

        let (node, prefix_len) = self.walk(input_ids)?;
        if prefix_len < input_ids.len() {
            let mut new_node = RadixNode::new(self.alloc_node_id(), Self::now_tick());
            new_node.key = input_ids[prefix_len..].to_vec();
            new_node.value = indices[prefix_len..].to_vec();
            new_node.parent = Some(Rc::downgrade(&node));
            let new_node_ref = Rc::new(RefCell::new(new_node));
            let edge = input_ids[prefix_len];
            node.borrow_mut().children.insert(edge, new_node_ref.clone());
            self.evictable_size += new_node_ref.borrow().len();
        }

        Ok(prefix_len)
    }

    fn evict(&mut self, size: usize) -> Result<Vec<i32>, CacheError> {
        if size == 0 {
            return Ok(Vec::new());
        }
        if size > self.evictable_size {
            return Err(CacheError::EvictTooLarge {
                requested: size,
                evictable: self.evictable_size,
            });
        }

        let mut heap = BinaryHeap::<Reverse<HeapEntry>>::new();
        for node in self.collect_leaf_nodes_for_evict() {
            let borrowed = node.borrow();
            heap.push(Reverse(HeapEntry {
                timestamp: borrowed.timestamp,
                node_id: borrowed.id,
                node: node.clone(),
            }));
        }

        let mut evicted_size = 0usize;
        let mut evicted_indices = Vec::new();

        while evicted_size < size {
            let Some(Reverse(entry)) = heap.pop() else {
                return Err(CacheError::CorruptedTree {
                    reason: "failed to evict enough cache",
                });
            };

            let node = entry.node;
            let (is_root, is_leaf, ref_count, node_len, node_value, edge) = {
                let borrowed = node.borrow();
                (
                    borrowed.is_root(),
                    borrowed.is_leaf(),
                    borrowed.ref_count,
                    borrowed.len(),
                    borrowed.value.clone(),
                    borrowed.key.first().copied(),
                )
            };
            if is_root || !is_leaf || ref_count > 0 {
                continue;
            }

            evicted_size += node_len;
            evicted_indices.extend(node_value);
            self.evictable_size =
                self.evictable_size
                    .checked_sub(node_len)
                    .ok_or(CacheError::CorruptedTree {
                        reason: "evictable_size underflow during eviction",
                    })?;

            let parent = Self::parent_of(&node)?;
            let edge = edge.ok_or(CacheError::CorruptedTree {
                reason: "evicted node has empty key",
            })?;
            parent.borrow_mut().children.remove(&edge);

            let should_push_parent = {
                let parent_borrow = parent.borrow();
                !parent_borrow.is_root() && parent_borrow.is_leaf() && parent_borrow.ref_count == 0
            };
            if should_push_parent {
                let parent_borrow = parent.borrow();
                heap.push(Reverse(HeapEntry {
                    timestamp: parent_borrow.timestamp,
                    node_id: parent_borrow.id,
                    node: parent.clone(),
                }));
            }
        }

        Ok(evicted_indices)
    }

    fn size_info(&self) -> SizeInfo {
        SizeInfo {
            evictable_size: self.evictable_size,
            protected_size: self.protected_size,
        }
    }

    fn check_integrity(&self) -> Result<(), CacheError> {
        if self.root_node.borrow().ref_count != 1 {
            return Err(CacheError::CorruptedTree {
                reason: "root ref_count must stay at 1",
            });
        }

        let mut stack = vec![self.root_node.clone()];
        let mut evictable_sum = 0usize;
        let mut protected_sum = 0usize;

        while let Some(node) = stack.pop() {
            let borrowed = node.borrow();
            let is_root = borrowed.is_root();

            if is_root {
                if !borrowed.key.is_empty() || !borrowed.value.is_empty() {
                    return Err(CacheError::CorruptedTree {
                        reason: "root key/value must be empty",
                    });
                }
            } else {
                if borrowed.key.is_empty() || borrowed.key.len() != borrowed.value.len() {
                    return Err(CacheError::CorruptedTree {
                        reason: "node key/value shape mismatch",
                    });
                }

                if borrowed.ref_count == 0 {
                    evictable_sum += borrowed.len();
                } else {
                    protected_sum += borrowed.len();
                }
            }

            for (edge, child) in &borrowed.children {
                let child_borrow = child.borrow();
                if child_borrow.key.first() != Some(edge) {
                    return Err(CacheError::CorruptedTree {
                        reason: "child edge key mismatch",
                    });
                }
                let child_parent = child_borrow
                    .parent
                    .as_ref()
                    .and_then(Weak::upgrade)
                    .ok_or(CacheError::CorruptedTree {
                        reason: "child parent pointer missing",
                    })?;
                if !Rc::ptr_eq(&child_parent, &node) {
                    return Err(CacheError::CorruptedTree {
                        reason: "child parent pointer mismatch",
                    });
                }
                drop(child_borrow);
                stack.push(child.clone());
            }
        }

        if evictable_sum != self.evictable_size || protected_sum != self.protected_size {
            return Err(CacheError::CorruptedTree {
                reason: "size accounting mismatch",
            });
        }
        Ok(())
    }
}
