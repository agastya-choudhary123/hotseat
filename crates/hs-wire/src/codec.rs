//! Hand-rolled little-endian encoding.
//!
//! No serialisation crate: the wire format is small, it has to be stable
//! forever (a sender and a receiver are different processes, possibly different
//! builds on different operating systems), and writing it out means the format
//! is readable in one file rather than inferred from derives.

pub struct Writer {
    pub buf: Vec<u8>,
}

impl Writer {
    pub fn new() -> Writer {
        Writer { buf: Vec::with_capacity(256) }
    }
    pub fn u8(&mut self, v: u8) -> &mut Self {
        self.buf.push(v);
        self
    }
    pub fn u32(&mut self, v: u32) -> &mut Self {
        self.buf.extend_from_slice(&v.to_le_bytes());
        self
    }
    pub fn u64(&mut self, v: u64) -> &mut Self {
        self.buf.extend_from_slice(&v.to_le_bytes());
        self
    }
    pub fn f32(&mut self, v: f32) -> &mut Self {
        // Bits, not a decimal rendering: a sampler temperature has to survive
        // the trip exactly or the two ends sample differently.
        self.buf.extend_from_slice(&v.to_bits().to_le_bytes());
        self
    }
    pub fn str(&mut self, s: &str) -> &mut Self {
        self.u32(s.len() as u32);
        self.buf.extend_from_slice(s.as_bytes());
        self
    }
    pub fn u32s(&mut self, v: &[u32]) -> &mut Self {
        self.u32(v.len() as u32);
        for &x in v {
            self.buf.extend_from_slice(&x.to_le_bytes());
        }
        self
    }
}

impl Default for Writer {
    fn default() -> Self {
        Writer::new()
    }
}

#[derive(Debug)]
pub struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}

#[derive(Debug)]
pub struct Truncated;

impl std::fmt::Display for Truncated {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "message ended early")
    }
}
impl std::error::Error for Truncated {}

type R<T> = Result<T, Truncated>;

impl<'a> Reader<'a> {
    pub fn new(b: &'a [u8]) -> Reader<'a> {
        Reader { b, p: 0 }
    }
    fn take(&mut self, n: usize) -> R<&'a [u8]> {
        if self.p + n > self.b.len() {
            return Err(Truncated);
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    pub fn u8(&mut self) -> R<u8> {
        Ok(self.take(1)?[0])
    }
    pub fn u32(&mut self) -> R<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub fn u64(&mut self) -> R<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    pub fn f32(&mut self) -> R<f32> {
        Ok(f32::from_bits(self.u32()?))
    }
    pub fn str(&mut self) -> R<String> {
        let n = self.u32()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    pub fn u32s(&mut self) -> R<Vec<u32>> {
        let n = self.u32()? as usize;
        let b = self.take(n * 4)?;
        Ok((0..n).map(|i| u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap())).collect())
    }
    pub fn remaining(&self) -> usize {
        self.b.len() - self.p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_every_type() {
        let mut w = Writer::new();
        w.u8(7).u32(0xDEAD_BEEF).u64(u64::MAX).f32(-0.125).str("hé").u32s(&[1, 2, 3]);
        let mut r = Reader::new(&w.buf);
        assert_eq!(r.u8().unwrap(), 7);
        assert_eq!(r.u32().unwrap(), 0xDEAD_BEEF);
        assert_eq!(r.u64().unwrap(), u64::MAX);
        assert_eq!(r.f32().unwrap(), -0.125);
        assert_eq!(r.str().unwrap(), "hé");
        assert_eq!(r.u32s().unwrap(), vec![1, 2, 3]);
        assert_eq!(r.remaining(), 0);
    }

    #[test]
    fn f32_survives_bit_for_bit() {
        for v in [0.8f32, 0.95, 1.1, f32::MIN_POSITIVE, -0.0] {
            let mut w = Writer::new();
            w.f32(v);
            assert_eq!(Reader::new(&w.buf).f32().unwrap().to_bits(), v.to_bits());
        }
    }

    #[test]
    fn short_input_is_an_error_not_a_panic() {
        let mut w = Writer::new();
        w.u32(1);
        let mut r = Reader::new(&w.buf[..3]);
        assert!(r.u32().is_err());
    }
}
