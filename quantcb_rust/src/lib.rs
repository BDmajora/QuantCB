use pyo3::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::BinaryHeap;
use std::cmp::Ordering;

// --- Data Structures ---

#[derive(Eq, PartialEq)]
struct StatsEntry {
    count: i32,
    pair: (u32, u32),
}

impl Ord for StatsEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.count.cmp(&other.count).then_with(|| self.pair.cmp(&other.pair))
    }
}

impl PartialOrd for StatsEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Debug)]
struct Node {
    val: u32,
    prev: i32,
    next: i32,
}

#[derive(Eq, PartialEq)]
struct ReverseRank(usize);

impl Ord for ReverseRank {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.cmp(&self.0)
    }
}

impl PartialOrd for ReverseRank {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// --- Functions ---

#[pyfunction]
fn train_bpe(text_bytes: &[u8], target_vocab_size: usize) -> PyResult<Vec<((u32, u32), u32)>> {
    if text_bytes.len() < 2 { return Ok(vec![]); }
    
    let mut list: Vec<Node> = text_bytes.iter().enumerate().map(|(i, &v)| Node {
        val: v as u32,
        prev: (i as i32) - 1,
        next: if i == text_bytes.len() - 1 { -1 } else { (i as i32) + 1 },
    }).collect();

    let mut stats = FxHashMap::default();
    for window in text_bytes.windows(2) {
        *stats.entry((window[0] as u32, window[1] as u32)).or_insert(0) += 1;
    }

    let mut pq: BinaryHeap<StatsEntry> = stats
        .iter()
        .map(|(&pair, &count)| StatsEntry { count, pair })
        .collect();

    let num_merges = target_vocab_size.saturating_sub(256);
    let mut merges = Vec::with_capacity(num_merges);

    for i in 0..num_merges {
        let mut best_pair = None;
        while let Some(entry) = pq.pop() {
            if let Some(&current_count) = stats.get(&entry.pair) {
                if current_count == entry.count {
                    best_pair = Some(entry.pair);
                    break;
                }
            }
        }

        let pair = match best_pair {
            Some(p) => p,
            None => break,
        };

        let new_id = 256 + i as u32;
        merges.push((pair, new_id));

        let mut curr = 0; 
        while curr != -1 {
            let idx = curr as usize;
            let next_idx = list[idx].next;

            if next_idx != -1 {
                let n_idx = next_idx as usize;
                if list[idx].val == pair.0 && list[n_idx].val == pair.1 {
                    if list[idx].prev != -1 {
                        let p_idx = list[idx].prev as usize;
                        let p_pair = (list[p_idx].val, list[idx].val);
                        *stats.entry(p_pair).or_insert(0) -= 1;
                        let new_p_pair = (list[p_idx].val, new_id);
                        let c = stats.entry(new_p_pair).or_insert(0);
                        *c += 1;
                        pq.push(StatsEntry { count: *c, pair: new_p_pair });
                    }
                    if list[n_idx].next != -1 {
                        let nn_idx = list[n_idx].next as usize;
                        let n_pair = (list[n_idx].val, list[nn_idx].val);
                        *stats.entry(n_pair).or_insert(0) -= 1;
                        let new_n_pair = (new_id, list[nn_idx].val);
                        let c = stats.entry(new_n_pair).or_insert(0);
                        *c += 1;
                        pq.push(StatsEntry { count: *c, pair: new_n_pair });
                    }

                    list[idx].val = new_id;
                    let after_next = list[n_idx].next;
                    list[idx].next = after_next;
                    if after_next != -1 {
                        list[after_next as usize].prev = idx as i32;
                    }
                }
            }
            curr = list[idx].next;
        }
        stats.remove(&pair);
    }
    Ok(merges)
}

#[pyfunction]
fn encode_bpe(tokens: Vec<u32>, merges: Vec<((u32, u32), u32)>) -> PyResult<Vec<u32>> {
    if tokens.len() < 2 { return Ok(tokens); }

    let mut vocab = FxHashMap::default();
    for (rank, &(pair, _)) in merges.iter().enumerate() {
        vocab.insert(pair, rank);
    }

    let mut list: Vec<Node> = tokens.iter().enumerate().map(|(i, &v)| Node {
        val: v,
        prev: (i as i32) - 1,
        next: if i == tokens.len() - 1 { -1 } else { (i as i32) + 1 },
    }).collect();

    let mut pq = BinaryHeap::new();
    
    let push_if_valid = |i: usize, l: &[Node], v: &FxHashMap<(u32, u32), usize>, q: &mut BinaryHeap<(ReverseRank, usize)>| {
        let n = l[i].next;
        if n != -1 {
            if let Some(&rank) = v.get(&(l[i].val, l[n as usize].val)) {
                q.push((ReverseRank(rank), i));
            }
        }
    };

    for i in 0..list.len() {
        push_if_valid(i, &list, &vocab, &mut pq);
    }

    while let Some((ReverseRank(rank), i)) = pq.pop() {
        if list[i].next == -1 { continue; }
        let j = list[i].next as usize;
        
        if let Some(&r) = vocab.get(&(list[i].val, list[j].val)) {
            if r != rank { continue; }

            let new_id = merges[rank].1;
            list[i].val = new_id;
            let next_next = list[j].next;
            list[i].next = next_next;
            if next_next != -1 {
                list[next_next as usize].prev = i as i32;
            }

            if list[i].prev != -1 {
                push_if_valid(list[i].prev as usize, &list, &vocab, &mut pq);
            }
            push_if_valid(i, &list, &vocab, &mut pq);
        }
    }

    let mut result = Vec::new();
    let mut curr = 0;
    while list[curr].prev != -1 { curr = list[curr].prev as usize; }
    let mut idx = Some(curr);
    while let Some(i) = idx {
        result.push(list[i].val);
        idx = if list[i].next == -1 { None } else { Some(list[i].next as usize) };
    }
    Ok(result)
}

#[pyfunction]
fn decode_bpe(tokens: Vec<u32>, merges: Vec<((u32, u32), u32)>) -> PyResult<String> {
    let mut vocab: FxHashMap<u32, Vec<u8>> = (0..256)
        .map(|i| (i as u32, vec![i as u8]))
        .collect();

    for ((p1, p2), new_id) in merges {
        let mut combined = vocab.get(&p1).cloned().unwrap_or_default();
        let mut second = vocab.get(&p2).cloned().unwrap_or_default();
        combined.append(&mut second);
        vocab.insert(new_id, combined);
    }

    let mut result_bytes = Vec::new();
    for token in tokens {
        if let Some(bytes) = vocab.get(&token) {
            result_bytes.extend_from_slice(bytes);
        }
    }

    String::from_utf8(result_bytes)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyUnicodeDecodeError, _>(e.to_string()))
}

#[pymodule]
fn quantcb_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(train_bpe, m)?)?;
    m.add_function(wrap_pyfunction!(encode_bpe, m)?)?;
    m.add_function(wrap_pyfunction!(decode_bpe, m)?)?;
    Ok(())
}