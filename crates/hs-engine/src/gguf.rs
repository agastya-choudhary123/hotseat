//! A GGUF v2/v3 reader.
//!
//! The file is mapped, never copied. Tensor data stays in the page cache and the
//! dequantiser reads straight out of it, so loading a second worker on the same
//! host costs no extra physical memory for the file itself.

use crate::mmap::Mmap;
use std::collections::HashMap;
use std::fmt;
use std::path::Path;

pub const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    BadMagic(u32),
    BadVersion(u32),
    Truncated { need: usize, have: usize },
    BadValueType(u32),
    BadTensorType(u32),
    MissingKey(String),
    WrongKeyType { key: String, want: &'static str },
    MissingTensor(String),
    TensorOutOfBounds { name: String },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "io: {e}"),
            Error::BadMagic(m) => write!(f, "not a GGUF file (magic {m:#010x})"),
            Error::BadVersion(v) => write!(f, "unsupported GGUF version {v}"),
            Error::Truncated { need, have } => {
                write!(f, "file truncated: needed {need} bytes, have {have}")
            }
            Error::BadValueType(t) => write!(f, "unknown metadata value type {t}"),
            Error::BadTensorType(t) => write!(f, "unknown tensor type {t}"),
            Error::MissingKey(k) => write!(f, "missing metadata key {k:?}"),
            Error::WrongKeyType { key, want } => write!(f, "key {key:?} is not {want}"),
            Error::MissingTensor(n) => write!(f, "missing tensor {n:?}"),
            Error::TensorOutOfBounds { name } => write!(f, "tensor {name:?} runs past end of file"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

type Result<T> = std::result::Result<T, Error>;

/// GGML tensor element types. Only the ones this project loads are named; the
/// rest are rejected at load time rather than silently mis-read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    Q5_0,
    Q8_0,
    Q4K,
    Q5K,
    Q6K,
}

impl GgmlType {
    fn from_u32(v: u32) -> Result<Self> {
        Ok(match v {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            6 => GgmlType::Q5_0,
            8 => GgmlType::Q8_0,
            12 => GgmlType::Q4K,
            13 => GgmlType::Q5K,
            14 => GgmlType::Q6K,
            other => return Err(Error::BadTensorType(other)),
        })
    }

    /// (elements per block, bytes per block)
    pub const fn block(self) -> (usize, usize) {
        match self {
            GgmlType::F32 => (1, 4),
            GgmlType::F16 => (1, 2),
            GgmlType::Q5_0 => (32, 22),
            GgmlType::Q8_0 => (32, 34),
            GgmlType::Q4K => (256, 144),
            GgmlType::Q5K => (256, 176),
            GgmlType::Q6K => (256, 210),
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            GgmlType::F32 => "F32",
            GgmlType::F16 => "F16",
            GgmlType::Q5_0 => "Q5_0",
            GgmlType::Q8_0 => "Q8_0",
            GgmlType::Q4K => "Q4_K",
            GgmlType::Q5K => "Q5_K",
            GgmlType::Q6K => "Q6_K",
        }
    }
}

#[derive(Debug, Clone)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    Str(String),
    /// Arrays of scalars are kept as a byte range into the mapping so that a
    /// 151936-entry vocabulary does not cost a Vec of Strings until asked for.
    Array { ty: u32, len: usize, off: usize },
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    /// Absolute file offset of the tensor's first byte.
    pub off: usize,
    /// Length in bytes.
    pub len: usize,
}

impl TensorInfo {
    pub fn elems(&self) -> usize {
        self.dims.iter().product::<u64>() as usize
    }
}

pub struct Gguf {
    map: Mmap,
    pub version: u32,
    pub meta: HashMap<String, Value>,
    pub tensors: HashMap<String, TensorInfo>,
    pub tensor_order: Vec<String>,
}

/// Sequential little-endian reader over the mapping.
struct Cur<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Cur<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.p + n > self.b.len() {
            return Err(Error::Truncated { need: self.p + n, have: self.b.len() });
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    fn value(&mut self, ty: u32) -> Result<Value> {
        Ok(match ty {
            0 => Value::U8(self.take(1)?[0]),
            1 => Value::I8(self.take(1)?[0] as i8),
            2 => Value::U16(u16::from_le_bytes(self.take(2)?.try_into().unwrap())),
            3 => Value::I16(i16::from_le_bytes(self.take(2)?.try_into().unwrap())),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_bits(self.u32()?)),
            7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::Str(self.string()?),
            9 => {
                let ety = self.u32()?;
                let len = self.u64()? as usize;
                let off = self.p;
                // Skip the payload. Strings are variable length, so walk them.
                if ety == 8 {
                    for _ in 0..len {
                        let n = self.u64()? as usize;
                        self.take(n)?;
                    }
                } else {
                    let w = scalar_width(ety)?;
                    self.take(len * w)?;
                }
                Value::Array { ty: ety, len, off }
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_bits(self.u64()?)),
            other => return Err(Error::BadValueType(other)),
        })
    }
}

fn scalar_width(ty: u32) -> Result<usize> {
    Ok(match ty {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4 | 5 | 6 => 4,
        10 | 11 | 12 => 8,
        other => return Err(Error::BadValueType(other)),
    })
}

impl Gguf {
    pub fn open(path: &Path) -> Result<Gguf> {
        let map = Mmap::open(path)?;
        let b = map.as_slice();
        let mut c = Cur { b, p: 0 };

        let magic = c.u32()?;
        if magic != GGUF_MAGIC {
            return Err(Error::BadMagic(magic));
        }
        let version = c.u32()?;
        if version != 2 && version != 3 {
            return Err(Error::BadVersion(version));
        }
        let n_tensors = c.u64()? as usize;
        let n_kv = c.u64()? as usize;

        let mut meta = HashMap::with_capacity(n_kv);
        for _ in 0..n_kv {
            let key = c.string()?;
            let ty = c.u32()?;
            let val = c.value(ty)?;
            meta.insert(key, val);
        }

        // Tensor descriptors, then padding to `general.alignment`, then data.
        let mut raw = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = c.string()?;
            let nd = c.u32()? as usize;
            let mut dims = Vec::with_capacity(nd);
            for _ in 0..nd {
                dims.push(c.u64()?);
            }
            let ty = GgmlType::from_u32(c.u32()?)?;
            let off = c.u64()? as usize;
            raw.push((name, dims, ty, off));
        }

        let align = match meta.get("general.alignment") {
            Some(Value::U32(a)) => *a as usize,
            _ => 32,
        };
        let data_start = (c.p + align - 1) / align * align;

        let mut tensors = HashMap::with_capacity(n_tensors);
        let mut tensor_order = Vec::with_capacity(n_tensors);
        for (name, dims, ty, rel) in raw {
            let elems: u64 = dims.iter().product();
            let (blk_elems, blk_bytes) = ty.block();
            let len = (elems as usize / blk_elems) * blk_bytes;
            let off = data_start + rel;
            if off + len > b.len() {
                return Err(Error::TensorOutOfBounds { name });
            }
            tensor_order.push(name.clone());
            tensors.insert(name.clone(), TensorInfo { name, dims, ty, off, len });
        }

        Ok(Gguf { map, version, meta, tensors, tensor_order })
    }

    pub fn bytes(&self) -> &[u8] {
        self.map.as_slice()
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors.get(name).ok_or_else(|| Error::MissingTensor(name.to_string()))
    }

    pub fn tensor_bytes(&self, t: &TensorInfo) -> &[u8] {
        &self.map.as_slice()[t.off..t.off + t.len]
    }

    pub fn get_u32(&self, key: &str) -> Result<u32> {
        match self.meta.get(key) {
            Some(Value::U32(v)) => Ok(*v),
            Some(Value::I32(v)) => Ok(*v as u32),
            Some(Value::U64(v)) => Ok(*v as u32),
            Some(_) => Err(Error::WrongKeyType { key: key.into(), want: "an integer" }),
            None => Err(Error::MissingKey(key.into())),
        }
    }

    pub fn get_f32(&self, key: &str) -> Result<f32> {
        match self.meta.get(key) {
            Some(Value::F32(v)) => Ok(*v),
            Some(Value::F64(v)) => Ok(*v as f32),
            Some(_) => Err(Error::WrongKeyType { key: key.into(), want: "a float" }),
            None => Err(Error::MissingKey(key.into())),
        }
    }

    pub fn get_str(&self, key: &str) -> Result<&str> {
        match self.meta.get(key) {
            Some(Value::Str(s)) => Ok(s),
            Some(_) => Err(Error::WrongKeyType { key: key.into(), want: "a string" }),
            None => Err(Error::MissingKey(key.into())),
        }
    }

    /// Materialise a string array (vocabulary, merge list).
    pub fn get_str_array(&self, key: &str) -> Result<Vec<String>> {
        let (len, off) = match self.meta.get(key) {
            Some(Value::Array { ty: 8, len, off }) => (*len, *off),
            Some(_) => Err(Error::WrongKeyType { key: key.into(), want: "a string array" })?,
            None => return Err(Error::MissingKey(key.into())),
        };
        let mut c = Cur { b: self.map.as_slice(), p: off };
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            out.push(c.string()?);
        }
        Ok(out)
    }

    pub fn get_i32_array(&self, key: &str) -> Result<Vec<i32>> {
        let (len, off) = match self.meta.get(key) {
            Some(Value::Array { ty: 5, len, off }) => (*len, *off),
            Some(_) => Err(Error::WrongKeyType { key: key.into(), want: "an i32 array" })?,
            None => return Err(Error::MissingKey(key.into())),
        };
        let b = self.map.as_slice();
        Ok((0..len)
            .map(|i| i32::from_le_bytes(b[off + i * 4..off + i * 4 + 4].try_into().unwrap()))
            .collect())
    }
}
