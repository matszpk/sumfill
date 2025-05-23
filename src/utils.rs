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

pub struct CombineIterStd {
    comb_iter: CombineIter,
    first: bool,
}

impl CombineIterStd {
    pub fn new(len: usize, values: usize) -> CombineIterStd {
        CombineIterStd {
            comb_iter: CombineIter::new(len, values),
            first: true,
        }
    }
}

impl Iterator for CombineIterStd {
    type Item = Vec<usize>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.first {
            self.first = false;
        } else if !self.comb_iter.next() {
            return None;
        }
        Some(Vec::<_>::from(self.comb_iter.get()))
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

pub fn combinations(len: u64, values: u64) -> u64 {
    if len > values {
        return 0;
    }
    let diff = values - len;
    let (diff, len) = if diff < len {
        (diff, len)
    } else {
        (len, diff)
    };
    let mut product: u64 = 1;
    let mut divider = 2;
    for f in len + 1..=values {
        //product = product.checked_mul(f).unwrap();
        if let Some(p) = product.checked_mul(f) {
            product = p;
            while divider <= diff && (product % divider) == 0 {
                product /= divider;
                divider += 1;
            }
        } else {
            panic!("comberr: {} {}", len, values);
        }
    }
    product
}
