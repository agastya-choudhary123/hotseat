//! Byte-level BPE, built from the vocabulary and merge table inside the GGUF.
//!
//! Written out rather than pulled from a crate for the same reason as the math:
//! the two ends of a migration have to agree exactly, and a tokenizer that
//! depends on a regex engine's version-to-version behaviour is one more thing
//! that can differ between two hosts. The Qwen2 pre-tokenizer pattern is
//! implemented directly as a scanner, alternative by alternative, in the order
//! a leftmost-first regex engine would try them.

use crate::gguf::Gguf;
use std::collections::HashMap;

pub struct Tokenizer {
    vocab: Vec<String>,
    ids: HashMap<String, u32>,
    ranks: HashMap<(String, String), u32>,
    /// byte -> the printable char GPT-2 byte-encoding maps it to
    b2u: [char; 256],
    /// and back
    u2b: HashMap<char, u8>,
    pub bos: Option<u32>,
    pub eos: u32,
    pub specials: Vec<(String, u32)>,
}

/// GPT-2's byte-to-unicode table: every byte gets a printable codepoint so that
/// BPE can work on strings without ever meeting a control character.
fn byte_to_unicode() -> ([char; 256], HashMap<char, u8>) {
    let mut b2u = ['\0'; 256];
    let mut used = [false; 256];
    let mut direct = Vec::new();
    direct.extend(b'!'..=b'~');
    direct.extend(0xA1u8..=0xAC);
    direct.extend(0xAEu8..=0xFF);
    for b in direct {
        b2u[b as usize] = b as char;
        used[b as usize] = true;
    }
    let mut n = 0u32;
    for b in 0..256usize {
        if !used[b] {
            b2u[b] = char::from_u32(256 + n).unwrap();
            n += 1;
        }
    }
    let mut u2b = HashMap::with_capacity(256);
    for b in 0..256usize {
        u2b.insert(b2u[b], b as u8);
    }
    (b2u, u2b)
}

impl Tokenizer {
    pub fn from_gguf(g: &Gguf) -> Result<Tokenizer, crate::gguf::Error> {
        let vocab = g.get_str_array("tokenizer.ggml.tokens")?;
        let merges = g.get_str_array("tokenizer.ggml.merges")?;
        let types = g.get_i32_array("tokenizer.ggml.token_type").unwrap_or_default();

        let mut ids = HashMap::with_capacity(vocab.len());
        for (i, t) in vocab.iter().enumerate() {
            ids.entry(t.clone()).or_insert(i as u32);
        }
        let mut ranks = HashMap::with_capacity(merges.len());
        for (r, m) in merges.iter().enumerate() {
            // Merge entries are "left right"; the left half may itself contain
            // no spaces because spaces are byte-encoded to 'Ġ'.
            if let Some(sp) = m.find(' ') {
                ranks.insert((m[..sp].to_string(), m[sp + 1..].to_string()), r as u32);
            }
        }
        // Control tokens (type 3) are matched literally in the input rather than
        // being reachable through BPE.
        let mut specials = Vec::new();
        for (i, t) in vocab.iter().enumerate() {
            let is_control = types.get(i).copied().unwrap_or(1) == 3;
            if is_control && t.starts_with("<|") {
                specials.push((t.clone(), i as u32));
            }
        }
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len())); // longest match first

        let (b2u, u2b) = byte_to_unicode();
        let eos = g.get_u32("tokenizer.ggml.eos_token_id")?;
        let bos = match g.get_u32("tokenizer.ggml.bos_token_id") {
            Ok(v) => {
                let add = matches!(g.meta.get("tokenizer.ggml.add_bos_token"),
                    Some(crate::gguf::Value::Bool(true)));
                if add { Some(v) } else { None }
            }
            Err(_) => None,
        };
        Ok(Tokenizer { vocab, ids, ranks, b2u, u2b, bos, eos, specials })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn id_of(&self, s: &str) -> Option<u32> {
        self.ids.get(s).copied()
    }

    /// Encode text, recognising control tokens like `<|im_start|>` literally.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        if let Some(b) = self.bos {
            out.push(b);
        }
        let mut rest = text;
        while !rest.is_empty() {
            // Find the earliest special token occurrence.
            let mut hit: Option<(usize, &str, u32)> = None;
            for (s, id) in &self.specials {
                if let Some(at) = rest.find(s.as_str()) {
                    if hit.map_or(true, |(h, _, _)| at < h) {
                        hit = Some((at, s, *id));
                    }
                }
            }
            match hit {
                Some((at, s, id)) => {
                    if at > 0 {
                        self.encode_ordinary(&rest[..at], &mut out);
                    }
                    out.push(id);
                    rest = &rest[at + s.len()..];
                }
                None => {
                    self.encode_ordinary(rest, &mut out);
                    break;
                }
            }
        }
        out
    }

    fn encode_ordinary(&self, text: &str, out: &mut Vec<u32>) {
        for piece in pretokenize(text) {
            let mut sym: Vec<String> =
                piece.bytes().map(|b| self.b2u[b as usize].to_string()).collect();
            self.bpe(&mut sym);
            for s in sym {
                match self.ids.get(&s) {
                    Some(&id) => out.push(id),
                    // Byte-level BPE guarantees every single-byte symbol is in
                    // the vocabulary, so this can only fire on a corrupt model.
                    None => panic!("token {s:?} is not in the vocabulary"),
                }
            }
        }
    }

    fn bpe(&self, sym: &mut Vec<String>) {
        loop {
            let mut best: Option<(u32, usize)> = None;
            for i in 0..sym.len().saturating_sub(1) {
                if let Some(&r) = self.ranks.get(&(sym[i].clone(), sym[i + 1].clone())) {
                    if best.map_or(true, |(br, _)| r < br) {
                        best = Some((r, i));
                    }
                }
            }
            let Some((_, i)) = best else { return };
            let merged = format!("{}{}", sym[i], sym[i + 1]);
            sym[i] = merged;
            sym.remove(i + 1);
        }
    }

    /// Decode ids back to bytes. Returns bytes, not a String, because a token
    /// boundary can fall inside a UTF-8 sequence.
    pub fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        for &id in ids {
            let Some(t) = self.vocab.get(id as usize) else { continue };
            for c in t.chars() {
                match self.u2b.get(&c) {
                    Some(&b) => out.push(b),
                    // A control token like <|im_end|> has no byte encoding;
                    // emit it verbatim so transcripts stay readable.
                    None => out.extend(c.to_string().as_bytes()),
                }
            }
        }
        out
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids)).into_owned()
    }

    pub fn token_text(&self, id: u32) -> String {
        self.decode(&[id])
    }
}

#[inline]
fn is_letter(c: char) -> bool {
    c.is_alphabetic()
}
#[inline]
fn is_number(c: char) -> bool {
    c.is_numeric()
}
#[inline]
fn is_nl(c: char) -> bool {
    c == '\r' || c == '\n'
}

/// The Qwen2 pre-tokenizer:
///
/// ```text
/// (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}
///   | ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
/// ```
///
/// Alternatives are tried in order at each position, and each is greedy with
/// backtracking — the two places that matters are spelled out below.
pub fn pretokenize(text: &str) -> Vec<String> {
    let c: Vec<char> = text.chars().collect();
    let n = c.len();
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let start = i;

        // 1. contractions, case-insensitive
        if c[i] == '\'' && i + 1 < n {
            let two: String = c[i + 1..(i + 3).min(n)].iter().collect::<String>().to_lowercase();
            let one = two.chars().next().unwrap();
            let len = if two.starts_with("re") || two.starts_with("ve") || two.starts_with("ll") {
                3
            } else if matches!(one, 's' | 't' | 'm' | 'd') {
                2
            } else {
                0
            };
            if len > 0 {
                out.push(c[i..i + len].iter().collect());
                i += len;
                continue;
            }
        }

        // 2. [^\r\n\p{L}\p{N}]?\p{L}+
        {
            let mut j = i;
            if !is_nl(c[j]) && !is_letter(c[j]) && !is_number(c[j]) {
                j += 1; // greedily take the optional leading char...
            }
            if j < n && is_letter(c[j]) {
                while j < n && is_letter(c[j]) {
                    j += 1;
                }
                out.push(c[i..j].iter().collect());
                i = j;
                continue;
            }
            // ...and if no letter followed, backtrack to not taking it.
            if is_letter(c[i]) {
                let mut j = i;
                while j < n && is_letter(c[j]) {
                    j += 1;
                }
                out.push(c[i..j].iter().collect());
                i = j;
                continue;
            }
        }

        // 3. \p{N} — one digit at a time
        if is_number(c[i]) {
            out.push(c[i].to_string());
            i += 1;
            continue;
        }

        // 4.  ?[^\s\p{L}\p{N}]+[\r\n]*
        {
            let mut j = i;
            if c[j] == ' ' {
                j += 1;
            }
            let sym_start = j;
            while j < n && !c[j].is_whitespace() && !is_letter(c[j]) && !is_number(c[j]) {
                j += 1;
            }
            if j > sym_start {
                while j < n && is_nl(c[j]) {
                    j += 1;
                }
                out.push(c[i..j].iter().collect());
                i = j;
                continue;
            }
        }

        // The rest all need whitespace here.
        if c[i].is_whitespace() {
            let mut run = i;
            while run < n && c[run].is_whitespace() {
                run += 1;
            }

            // 5. \s*[\r\n]+ — the longest \s* prefix that still leaves a newline.
            let mut k = run;
            loop {
                if k < n && is_nl(c[k]) {
                    let mut e = k;
                    while e < n && is_nl(c[e]) {
                        e += 1;
                    }
                    out.push(c[i..e].iter().collect());
                    i = e;
                    break;
                }
                if k == i {
                    break;
                }
                k -= 1;
            }
            if i != start {
                continue;
            }

            // 6. \s+(?!\S) — the run keeps its last character only if nothing
            //    follows it; otherwise that character belongs to the next word.
            if run == n {
                out.push(c[i..run].iter().collect());
                i = run;
                continue;
            }
            if run - i > 1 {
                out.push(c[i..run - 1].iter().collect());
                i = run - 1;
                continue;
            }

            // 7. \s+
            out.push(c[i..run].iter().collect());
            i = run;
            continue;
        }

        // Nothing matched (a lone control character); emit it so we always make
        // progress rather than looping forever.
        out.push(c[i].to_string());
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> Vec<String> {
        pretokenize(s)
    }

    #[test]
    fn splits_words_with_their_leading_space() {
        assert_eq!(p("Hello world"), vec!["Hello", " world"]);
        assert_eq!(p("the quick brown fox"), vec!["the", " quick", " brown", " fox"]);
    }

    #[test]
    fn splits_digits_individually() {
        assert_eq!(p("x=1234"), vec!["x", "=", "1", "2", "3", "4"]);
    }

    #[test]
    fn handles_contractions() {
        assert_eq!(p("don't"), vec!["don", "'t"]);
        assert_eq!(p("They'RE here"), vec!["They", "'RE", " here"]);
        assert_eq!(p("I'll've"), vec!["I", "'ll", "'ve"]);
    }

    #[test]
    fn trailing_run_keeps_all_its_spaces_only_at_the_end() {
        assert_eq!(p("a   "), vec!["a", "   "]);
        assert_eq!(p("a   b"), vec!["a", "  ", " b"]);
    }

    #[test]
    fn newlines_absorb_preceding_whitespace() {
        assert_eq!(p("a  \n\nb"), vec!["a", "  \n\n", "b"]);
        assert_eq!(p("a\n  b"), vec!["a", "\n", " ", " b"]);
    }

    #[test]
    fn punctuation_runs_stay_together() {
        assert_eq!(p("wow!!! ok"), vec!["wow", "!!!", " ok"]);
        // Alternative 2 pulls a single leading symbol into the following word,
        // which is why "(x" is one piece but ")" stands alone.
        assert_eq!(p("f(x) -> y"), vec!["f", "(x", ")", " ->", " y"]);
        assert_eq!(p("a.b"), vec!["a", ".b"]);
    }

    #[test]
    fn every_character_is_accounted_for() {
        for s in ["", "a", "  ", "\n", "\r\n\r\n", "héllo wörld 42!", "  leading", "tab\there"] {
            assert_eq!(p(s).concat(), s, "{s:?}");
        }
    }

    #[test]
    fn byte_map_is_a_bijection() {
        let (b2u, u2b) = byte_to_unicode();
        assert_eq!(u2b.len(), 256);
        for b in 0..256usize {
            assert_eq!(u2b[&b2u[b]], b as u8);
        }
        assert_eq!(b2u[b' ' as usize], 'Ġ');
        assert_eq!(b2u[b'\n' as usize], 'Ċ');
    }
}
