pub struct CombineIter {
    values: usize,
    item: Vec<usize>,
}

impl CombineIter {
    // len: k, values: n
    pub fn new(len: usize, values: usize) -> CombineIter {
        assert!(len <= values);
        CombineIter {
            values,
            item: (0..len).collect::<Vec<_>>(),
        }
    }

    pub fn next(&mut self) -> bool {
        let len: usize = self.item.len();
        for ki in 0..len {
            let i = len - ki - 1;
            self.item[i] += 1;
            if self.item[i] < self.values - ki {
                for j in 1..=ki {
                    self.item[i + j] = self.item[i] + j;
                }
                return true;
            }
        }
        false
    }

    pub fn get(&self) -> &[usize] {
        self.item.as_slice()
    }

    #[allow(dead_code)]
    pub fn values(&self) -> usize {
        self.values
    }
}

pub struct CombineWithRepIter {
    values: usize,
    item: Vec<usize>,
}

impl CombineWithRepIter {
    // len: k, values: n
    pub fn new(len: usize, values: usize) -> CombineWithRepIter {
        assert!(len <= values);
        CombineWithRepIter {
            values,
            item: vec![0; len],
        }
    }

    pub fn next(&mut self) -> bool {
        let len: usize = self.item.len();
        for ki in 0..len {
            let i = len - ki - 1;
            self.item[i] += 1;
            if self.item[i] < self.values {
                for j in 1..=ki {
                    self.item[i + j] = self.item[i];
                }
                return true;
            }
        }
        false
    }

    pub fn get(&self) -> &[usize] {
        self.item.as_slice()
    }

    #[allow(dead_code)]
    pub fn values(&self) -> usize {
        self.values
    }
}
