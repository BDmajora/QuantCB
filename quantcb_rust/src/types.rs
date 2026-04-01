use std::cmp::Ordering;

#[derive(Eq, PartialEq)]
pub struct StatsEntry {
    pub count: i32,
    pub pair: (u32, u32),
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
pub struct Node {
    pub val: u32,
    pub prev: i32,
    pub next: i32,
}

#[derive(Eq, PartialEq)]
pub struct ReverseRank(pub usize);

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