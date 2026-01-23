//! Convert spoken punctuation words to their symbols

/// Process text to convert spoken punctuation to symbols
///
/// Note: String slicing is safe here because all search terms are ASCII,
/// so found positions are guaranteed to be valid UTF-8 boundaries.
#[allow(clippy::string_slice)]
pub fn process(text: &str) -> String {
    let mut result = text.to_string();

    // Order matters - process longer phrases first
    let replacements = [
        // Redundant cases (word + symbol already present)
        ("period.", "."),
        ("comma,", ","),
        // Question/exclamation
        ("question mark", "?"),
        ("exclamation point", "!"),
        ("exclamation mark", "!"),
        // Basic punctuation
        ("period", "."),
        ("full stop", "."),
        ("comma", ","),
        ("colon", ":"),
        ("semicolon", ";"),
        ("semi colon", ";"),
        // Quotes and brackets
        ("open quote", "\""),
        ("close quote", "\""),
        ("quote", "\""),
        ("open paren", "("),
        ("close paren", ")"),
        ("open bracket", "["),
        ("close bracket", "]"),
        // Other
        ("hyphen", "-"),
        ("dash", "-"),
        ("ellipsis", "..."),
        ("ampersand", "&"),
        ("at sign", "@"),
        ("hashtag", "#"),
        ("dollar sign", "$"),
        ("percent sign", "%"),
        ("percent", "%"),
    ];

    for (spoken, symbol) in replacements {
        // Case-insensitive search and replace
        loop {
            let lower = result.to_lowercase();
            let Some(pos) = lower.find(spoken) else {
                break;
            };
            let end = pos + spoken.len();

            // Check if at word boundary (followed by space, punctuation, or end)
            let at_boundary = match result.chars().nth(end) {
                None => true,
                Some(c) => c.is_whitespace() || c.is_ascii_punctuation(),
            };

            if !at_boundary {
                break;
            }

            // Build replacement: symbol + space (unless at end or followed by punctuation)
            let suffix = match result.chars().nth(end) {
                None => "",
                Some(c) if c.is_whitespace() => " ",
                Some(_) => "", // punctuation follows, no space needed
            };

            // Skip whitespace after the spoken word
            let skip_end = if result.get(end..end + 1) == Some(" ") {
                end + 1
            } else {
                end
            };

            result = format!("{}{}{}{}", &result[..pos], symbol, suffix, &result[skip_end..]);
        }
    }

    // Clean up spacing around punctuation (remove space before)
    result = result
        .replace(" .", ".")
        .replace(" ,", ",")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" :", ":")
        .replace(" ;", ";");

    // Fix double spaces
    while result.contains("  ") {
        result = result.replace("  ", " ");
    }

    result.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_punctuation() {
        assert_eq!(process("hello comma world"), "hello, world");
        assert_eq!(process("end of sentence period"), "end of sentence.");
        assert_eq!(process("what question mark"), "what?");
    }

    #[test]
    fn test_redundant_punctuation() {
        assert_eq!(process("end period."), "end.");
    }
}
