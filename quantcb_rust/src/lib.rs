use pyo3::prelude::*;
use std::collections::HashMap;

#[pyfunction]
fn train_bpe(text_bytes: &[u8], target_vocab_size: usize) -> PyResult<Vec<((u32, u32), u32)>> {
    let mut tokens: Vec<u32> = text_bytes.iter().map(|&b| b as u32).collect();
    let num_merges = target_vocab_size.saturating_sub(256);
    let mut merges = Vec::with_capacity(num_merges);

    for i in 0..num_merges {
        let mut stats: HashMap<(u32, u32), usize> = HashMap::new();
        for window in tokens.windows(2) {
            *stats.entry((window[0], window[1])).or_insert(0) += 1;
        }

        if stats.is_empty() { break; }

        // Find the pair with the highest frequency
        let best_pair = stats
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(pair, _)| pair)
            .unwrap();

        let new_id = 256 + i as u32;
        merges.push((best_pair, new_id));

        // Perform the merge across the current token list
        let mut new_tokens = Vec::with_capacity(tokens.len());
        let mut idx = 0;
        while idx < tokens.len() {
            if idx < tokens.len() - 1 && (tokens[idx], tokens[idx + 1]) == best_pair {
                new_tokens.push(new_id);
                idx += 2;
            } else {
                new_tokens.push(tokens[idx]);
                idx += 1;
            }
        }
        tokens = new_tokens;
    }
    Ok(merges)
}

#[pyfunction]
fn encode_bpe(mut tokens: Vec<u32>, merges: HashMap<(u32, u32), u32>) -> PyResult<Vec<u32>> {
    // Apply merges in the order they were learned by sorting by the new_id
    let mut sorted_merges: Vec<_> = merges.into_iter().collect();
    sorted_merges.sort_by_key(|&(_, new_id)| new_id);

    for (pair, new_id) in sorted_merges {
        let mut new_tokens = Vec::with_capacity(tokens.len());
        let mut i = 0;
        while i < tokens.len() {
            if i < tokens.len() - 1 && (tokens[i], tokens[i + 1]) == pair {
                new_tokens.push(new_id);
                i += 2;
            } else {
                new_tokens.push(tokens[i]);
                i += 1;
            }
        }
        tokens = new_tokens;
    }
    Ok(tokens)
}

/// The module definition using the modern PyO3 0.22+ Bound API
#[pymodule]
fn quantcb_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(train_bpe, m)?)?;
    m.add_function(wrap_pyfunction!(encode_bpe, m)?)?;
    Ok(())
}