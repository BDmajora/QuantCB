use pyo3::prelude::*;
use std::collections::HashMap;

#[pyfunction]
fn train_bpe(text_bytes: &[u8], target_vocab_size: usize) -> PyResult<Vec<((u32, u32), u32)>> {
    let mut tokens: Vec<u32> = text_bytes.iter().map(|&b| b as u32).collect();
    
    // Pre-allocate a secondary buffer to avoid constant reallocation
    let mut next_tokens = Vec::with_capacity(tokens.len());
    
    let num_merges = target_vocab_size.saturating_sub(256);
    let mut merges = Vec::with_capacity(num_merges);

    for i in 0..num_merges {
        // Pre-size the hashmap to reduce rehashing overhead
        let mut stats = HashMap::with_capacity(tokens.len() / 2); 
        for window in tokens.windows(2) {
            stats.entry((window[0], window[1]))
                .and_modify(|c| *c += 1)
                .or_insert(1);
        }

        if stats.is_empty() { break; }

        let best_pair = stats
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(pair, _)| pair)
            .unwrap();

        let new_id = 256 + i as u32;
        merges.push((best_pair, new_id));

        next_tokens.clear();
        let mut idx = 0;
        
        while idx < tokens.len() {
            if idx < tokens.len() - 1 && tokens[idx] == best_pair.0 && tokens[idx + 1] == best_pair.1 {
                next_tokens.push(new_id);
                idx += 2;
            } else {
                next_tokens.push(tokens[idx]);
                idx += 1;
            }
        }
        
        // Swap buffers to reuse memory and avoid allocations
        std::mem::swap(&mut tokens, &mut next_tokens);
    }
    Ok(merges)
}

#[pyfunction]
fn encode_bpe(mut tokens: Vec<u32>, merges: Vec<((u32, u32), u32)>) -> PyResult<Vec<u32>> {
    if tokens.is_empty() || merges.is_empty() {
        return Ok(tokens);
    }

    // Allocate once
    let mut next_tokens = Vec::with_capacity(tokens.len());

    for (pair, new_id) in merges {
        // Fast path scanner
        // This vectorizes efficiently and skips the heavy loop if the pair is missing
        if !tokens.windows(2).any(|w| w[0] == pair.0 && w[1] == pair.1) {
            continue;
        }

        next_tokens.clear();
        let mut i = 0;
        
        while i < tokens.len() {
            if i < tokens.len() - 1 && tokens[i] == pair.0 && tokens[i + 1] == pair.1 {
                next_tokens.push(new_id);
                i += 2;
            } else {
                next_tokens.push(tokens[i]);
                i += 1;
            }
        }
        
        std::mem::swap(&mut tokens, &mut next_tokens);
    }
    
    Ok(tokens)
}

#[pymodule]
fn quantcb_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(train_bpe, m)?)?;
    m.add_function(wrap_pyfunction!(encode_bpe, m)?)?;
    Ok(())
}