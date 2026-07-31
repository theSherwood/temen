//! A minimal NIF s-expression reader — enough for Leng (`doc/leng-spec.md`) and the upstream NIF
//! syntax (`nifspec`): parenthesised nodes, atoms, `:symbol-definitions`, string literals, the
//! `.nif`/`.indexat` directives, and per-node `@lineinfo` suffixes (which we strip — position is
//! not needed to translate, only to diagnose, and that is a later concern).
//!
//! NIF is an almost-classical Lisp: a list's first atom is its **tag**. We keep atoms verbatim
//! except the `@lineinfo` suffix, which nimony appends to a tag/symbol (`add@2,g,file`) and which
//! carries no semantic weight for translation.

/// A parsed NIF node: either an atom (tag, symbol, number, string-with-quotes, or the lone `.`
/// Empty marker) or a parenthesised list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Node {
    Atom(String),
    List(Vec<Node>),
}

impl Node {
    /// The list's tag (first atom), or `None` for an atom or an empty/headless list.
    pub fn tag(&self) -> Option<&str> {
        match self {
            Node::List(items) => match items.first() {
                Some(Node::Atom(a)) => Some(a.as_str()),
                _ => None,
            },
            _ => None,
        }
    }
    /// The children after the tag, for a tagged list; empty otherwise.
    pub fn args(&self) -> &[Node] {
        match self {
            Node::List(items) if !items.is_empty() => &items[1..],
            _ => &[],
        }
    }
    pub fn as_atom(&self) -> Option<&str> {
        match self {
            Node::Atom(a) => Some(a.as_str()),
            _ => None,
        }
    }
    /// True for the lone `.` Empty marker (NIF's "no value here").
    pub fn is_empty_marker(&self) -> bool {
        matches!(self, Node::Atom(a) if a == ".")
    }
}

/// Parse a NIF document into a single root node. A document is a sequence of top-level forms;
/// leading `(.nif…)`/`(.indexat…)` directives are dropped and the **last** non-directive form (the
/// `(stmts …)` module) is returned. (Leng modules are a single `stmts`; a bare sequence is wrapped.)
pub fn parse(src: &str) -> Result<Node, String> {
    let mut p = Parser {
        b: src.as_bytes(),
        i: 0,
    };
    let mut forms = Vec::new();
    loop {
        p.skip_ws();
        if p.i >= p.b.len() {
            break;
        }
        let n = p.node()?;
        // Drop directives: a list whose tag begins with '.' (`.nif27`, `.indexat`, `.indexdone`).
        let is_directive = n.tag().map(|t| t.starts_with('.')).unwrap_or(false);
        if !is_directive {
            forms.push(n);
        }
    }
    match forms.len() {
        0 => Err("empty NIF document".into()),
        1 => Ok(forms.pop().unwrap()),
        // Multiple top-level forms with no enclosing (stmts …): wrap them so the translator sees a
        // uniform module. Real nimony Leng is a single (stmts …), so this is mostly for fixtures.
        _ => {
            let mut items = vec![Node::Atom("stmts".into())];
            items.extend(forms);
            Ok(Node::List(items))
        }
    }
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn skip_ws(&mut self) {
        while self.i < self.b.len() {
            let c = self.b[self.i];
            if c == b' ' || c == b'\t' || c == b'\r' || c == b'\n' {
                self.i += 1;
            } else {
                break;
            }
        }
    }

    fn node(&mut self) -> Result<Node, String> {
        self.skip_ws();
        if self.i >= self.b.len() {
            return Err("unexpected end of input".into());
        }
        match self.b[self.i] {
            b'(' => self.list(),
            b')' => Err("unexpected ')'".into()),
            b'"' => self.string(),
            b'\'' => self.char_lit(),
            _ => self.atom(),
        }
    }

    fn list(&mut self) -> Result<Node, String> {
        debug_assert_eq!(self.b[self.i], b'(');
        self.i += 1; // consume '('
        let mut items = Vec::new();
        loop {
            self.skip_ws();
            if self.i >= self.b.len() {
                return Err("unterminated list".into());
            }
            if self.b[self.i] == b')' {
                self.i += 1;
                return Ok(Node::List(items));
            }
            items.push(self.node()?);
        }
    }

    fn string(&mut self) -> Result<Node, String> {
        debug_assert_eq!(self.b[self.i], b'"');
        let start = self.i;
        self.i += 1;
        while self.i < self.b.len() {
            match self.b[self.i] {
                b'\\' => self.i += 2, // NIF escapes are `\xx` / `\c`; skip the next byte
                b'"' => {
                    self.i += 1;
                    // Keep the quotes so the translator can distinguish a string atom.
                    return Ok(Node::Atom(
                        String::from_utf8_lossy(&self.b[start..self.i]).into_owned(),
                    ));
                }
                _ => self.i += 1,
            }
        }
        Err("unterminated string literal".into())
    }

    /// A NIF **character literal** — `'0'`, `' '` (a literal space), `'\5C'` (a `\HH` hex escape).
    /// It must be lexed as a unit rather than via `atom()`, because a space char (`' '`) or any
    /// other delimiter inside the quotes would otherwise split the token. A char literal never
    /// contains an unescaped `'`, so scanning to the next `'` finds the true close; any trailing
    /// line-info suffix (`@…`/`~…`) after the close is then discarded like an atom's.
    fn char_lit(&mut self) -> Result<Node, String> {
        debug_assert_eq!(self.b[self.i], b'\'');
        let start = self.i;
        self.i += 1; // opening '
        while self.i < self.b.len() && self.b[self.i] != b'\'' {
            self.i += 1;
        }
        if self.i >= self.b.len() {
            return Err("unterminated char literal".into());
        }
        self.i += 1; // closing '
        let end_lit = self.i;
        // Skip a trailing line-info suffix (`@…`/`~…`) up to the next delimiter.
        while self.i < self.b.len() {
            let c = self.b[self.i];
            if c == b'(' || c == b')' || c == b' ' || c == b'\t' || c == b'\r' || c == b'\n' {
                break;
            }
            self.i += 1;
        }
        Ok(Node::Atom(
            String::from_utf8_lossy(&self.b[start..end_lit]).into_owned(),
        ))
    }

    fn atom(&mut self) -> Result<Node, String> {
        let start = self.i;
        while self.i < self.b.len() {
            let c = self.b[self.i];
            if c == b'(' || c == b')' || c == b' ' || c == b'\t' || c == b'\r' || c == b'\n' {
                break;
            }
            self.i += 1;
        }
        let raw = String::from_utf8_lossy(&self.b[start..self.i]).into_owned();
        // Strip the NIF **line-info** suffix. Every token (tag, symbol, number, even the `.` Empty
        // marker) may carry position info introduced by `@` (absolute: `@col,line,file`) or `~`
        // (a relative/negative delta: `a.0~2`, `(i~6,~4 64)`). Neither char can occur in the
        // semantic part: builtin tags are plain words, mangled symbols encode `~`/`@` away
        // (`tildeQ`/`atQ`, spec name-mangling table), and integer literals use `-`, not `~`. So
        // cutting at the first `@`/`~` yields exactly the token, discarding only position info.
        let cleaned = match raw.find(['@', '~']) {
            Some(i) if i > 0 => raw[..i].to_string(),
            _ => raw,
        };
        Ok(Node::Atom(cleaned))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_nested() {
        let n = parse("(add (i +64) a.0 3)").unwrap();
        assert_eq!(n.tag(), Some("add"));
        assert_eq!(n.args().len(), 3);
        assert_eq!(n.args()[2].as_atom(), Some("3"));
    }

    #[test]
    fn strips_lineinfo_and_directives() {
        let n = parse("(.nif27)(stmts@,1,foo.nim (ret 3))").unwrap();
        assert_eq!(n.tag(), Some("stmts"));
        assert_eq!(n.args()[0].tag(), Some("ret"));
    }

    #[test]
    fn space_char_literal_is_one_atom() {
        // `' '` is a space char; the lexer must not split it on the interior space.
        let n = parse("(call f x ' ')").unwrap();
        assert_eq!(n.args().len(), 3);
        assert_eq!(n.args()[2].as_atom(), Some("' '"));
    }

    #[test]
    fn char_literals_with_lineinfo_and_escapes() {
        let n = parse("(call f '0'@G '\\5C'@4 ' ')").unwrap();
        assert_eq!(n.args().len(), 4);
        assert_eq!(n.args()[1].as_atom(), Some("'0'"));
        assert_eq!(n.args()[2].as_atom(), Some("'\\5C'"));
        assert_eq!(n.args()[3].as_atom(), Some("' '"));
    }

    #[test]
    fn keeps_strings_whole() {
        let n = parse(r#"(data "a b\3A c")"#).unwrap();
        assert_eq!(n.args()[0].as_atom(), Some(r#""a b\3A c""#));
    }
}
