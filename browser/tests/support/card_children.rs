//! A playground card's `//// child: NAME.c` programs (#2219), split as `play.js`'s `splitChildren`
//! splits them: the text above the first marker is the card's program, and each marker starts a child
//! program named NAME without its `.c`, which the page grants to the card's run under that name.

/// The card's program and its children, `(name, source)` in order.
pub fn split(src: &str) -> (String, Vec<(String, String)>) {
    let mut parent = String::new();
    let mut children: Vec<(String, String)> = Vec::new();
    for line in src.split_inclusive('\n') {
        let name = line
            .trim()
            .strip_prefix("////")
            .map(str::trim_start)
            .filter(|r| r.get(..6).is_some_and(|p| p.eq_ignore_ascii_case("child:")))
            .map(|r| r[6..].trim())
            .filter(|n| !n.is_empty() && !n.contains(char::is_whitespace));
        match (name, children.last_mut()) {
            (Some(n), _) => {
                children.push((n.strip_suffix(".c").unwrap_or(n).to_string(), String::new()))
            }
            (None, Some((_, child))) => child.push_str(line),
            (None, None) => parent.push_str(line),
        }
    }
    (parent, children)
}
