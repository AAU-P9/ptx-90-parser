use crate::parser::Span;
use crate::Spanned;
use serde::Serialize;

/// A structured `// @META` annotation embedded in PTX output.
///
/// These annotations are emitted by the `ptx_meta.h` framework and carry
/// zero-cost metadata about kernels, parameters, tiles, etc.
///
/// Protocol format: `// @META[:<version>] <TAG> <fields...>`
#[derive(Debug, Clone, PartialEq, Spanned, Serialize)]
pub struct MetaDirective {
    /// Protocol version (e.g. `3` from `@META:3`), or `None` if no version.
    pub version: Option<u32>,
    /// The parsed tag/body of the annotation.
    pub tag: MetaTag,
    pub span: Span,
}

/// The tag (body) of a `// @META` annotation, corresponding to the tags
/// defined in `ptx_meta.h`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum MetaTag {
    /// `BEGIN_KERNEL <name>`
    BeginKernel { name: String },
    /// `END_KERNEL <name>`
    EndKernel { name: String },
    /// `PARAM <index> <name> <type> <role> [constraints...]`
    Param {
        index: u32,
        name: String,
        param_type: String,
        role: String,
        constraints: Vec<MetaConstraint>,
    },
    /// `TILE <dim> <size>`
    Tile { dim: String, size: u32 },
    /// `LAUNCH <block_x> <block_y> <block_z> <grid_expr>`
    Launch {
        block_x: u32,
        block_y: u32,
        block_z: u32,
        grid_expr: String,
    },
    /// `SHARED_MEM <name> <elem_type> <total_bytes>`
    SharedMem {
        name: String,
        elem_type: String,
        total_bytes: u32,
    },
    /// `LOOP <label> <min_iters> <max_iters> <is_unrolled>`
    Loop {
        label: String,
        min_iters: u32,
        max_iters: u32,
        is_unrolled: bool,
    },
    /// `LAYOUT <name> <order> <dims_expr>`
    Layout {
        name: String,
        order: String,
        dims: String,
    },
    /// `ASSUME <expr_description>`
    Assume { description: String },
    /// `CONST_TABLE <symbol_name> <num_entries>`
    ConstTable {
        symbol_name: String,
        num_entries: u32,
    },
    /// `CONST <symbol_name> <fields>`
    Const { symbol_name: String, fields: String },
    /// `CUSTOM <key> <value...>`
    Custom { key: String, value: String },
    /// `VERSION <n>` (legacy v2 protocol)
    Version { version: u32 },
    /// `KERNEL <name>` (legacy v2 protocol)
    Kernel { name: String },
    /// Any unrecognised tag — kept as raw text so nothing is lost.
    Unknown { raw: String },
}

/// A constraint on a `PARAM` annotation (e.g. `range=1:8192`, `align=128`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum MetaConstraint {
    Range {
        lo: String,
        hi: String,
    },
    Stride {
        value: String,
    },
    Multiple {
        value: String,
    },
    Align {
        value: String,
    },
    ReadOnly,
    WriteOnly,
    ReadWrite,
    NoAlias,
    /// Unrecognised constraint token.
    Other(String),
}

/// Parse the raw content string of a `// @META` comment into a [`MetaDirective`].
///
/// The input `raw` is everything after `// @META` with leading/trailing
/// whitespace already trimmed (this is what the lexer callback produces).
pub fn parse_meta_content(raw: &str, span: Span) -> MetaDirective {
    let (version, body) = parse_version_prefix(raw);
    let tag = parse_tag(body);
    MetaDirective { version, tag, span }
}

/// Strip an optional `:N ` version prefix, returning `(Some(N), rest)` or
/// `(None, input)`.
fn parse_version_prefix(s: &str) -> (Option<u32>, &str) {
    if let Some(rest) = s.strip_prefix(':') {
        // Find the end of the version number
        let end = rest
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(rest.len());
        if end > 0 {
            if let Ok(v) = rest[..end].parse::<u32>() {
                let body = rest[end..].trim_start();
                return (Some(v), body);
            }
        }
    }
    (None, s)
}

fn parse_tag(body: &str) -> MetaTag {
    let mut parts = body.splitn(2, char::is_whitespace);
    let tag_name = parts.next().unwrap_or("");
    let rest = parts.next().unwrap_or("").trim();

    match tag_name {
        "BEGIN_KERNEL" => MetaTag::BeginKernel {
            name: rest.to_string(),
        },
        "END_KERNEL" => MetaTag::EndKernel {
            name: rest.to_string(),
        },
        "PARAM" => parse_param(rest),
        "TILE" => parse_tile(rest),
        "LAUNCH" => parse_launch(rest),
        "SHARED_MEM" => parse_shared_mem(rest),
        "LOOP" => parse_loop(rest),
        "LAYOUT" => parse_layout(rest),
        "ASSUME" => MetaTag::Assume {
            description: rest.to_string(),
        },
        "CONST_TABLE" => parse_const_table(rest),
        "CONST" => parse_const(rest),
        "CUSTOM" => parse_custom(rest),
        "VERSION" => parse_version_tag(rest),
        "KERNEL" => MetaTag::Kernel {
            name: rest.to_string(),
        },
        // Legacy v2 tags
        "PARAM_RANGE" | "PARAM_TYPE" | "PARAM_ALIGN" | "TILE_SIZE" | "KERNEL_SHAPE" | "MEMORY"
        | "LOOP_BOUNDS" => MetaTag::Unknown {
            raw: body.to_string(),
        },
        _ => MetaTag::Unknown {
            raw: body.to_string(),
        },
    }
}

/// `PARAM <index> <name> <type> <role> [constraints...]`
fn parse_param(rest: &str) -> MetaTag {
    let tokens: Vec<&str> = rest.split_whitespace().collect();
    if tokens.len() < 4 {
        return MetaTag::Unknown {
            raw: format!("PARAM {}", rest),
        };
    }
    let index = tokens[0].parse::<u32>().unwrap_or(0);
    let name = tokens[1].to_string();
    let param_type = tokens[2].to_string();
    let role = tokens[3].to_string();
    let constraints = tokens[4..].iter().map(|t| parse_constraint(t)).collect();
    MetaTag::Param {
        index,
        name,
        param_type,
        role,
        constraints,
    }
}

fn parse_constraint(s: &str) -> MetaConstraint {
    if let Some(rest) = s.strip_prefix("range=") {
        let mut parts = rest.splitn(2, ':');
        let lo = parts.next().unwrap_or("").to_string();
        let hi = parts.next().unwrap_or("").to_string();
        MetaConstraint::Range { lo, hi }
    } else if let Some(rest) = s.strip_prefix("stride=") {
        MetaConstraint::Stride {
            value: rest.to_string(),
        }
    } else if let Some(rest) = s.strip_prefix("multiple=") {
        MetaConstraint::Multiple {
            value: rest.to_string(),
        }
    } else if let Some(rest) = s.strip_prefix("align=") {
        MetaConstraint::Align {
            value: rest.to_string(),
        }
    } else if s == "readonly" {
        MetaConstraint::ReadOnly
    } else if s == "writeonly" {
        MetaConstraint::WriteOnly
    } else if s == "readwrite" {
        MetaConstraint::ReadWrite
    } else if s == "noalias" {
        MetaConstraint::NoAlias
    } else {
        MetaConstraint::Other(s.to_string())
    }
}

/// `TILE <dim> <size>`
fn parse_tile(rest: &str) -> MetaTag {
    let tokens: Vec<&str> = rest.split_whitespace().collect();
    if tokens.len() >= 2 {
        MetaTag::Tile {
            dim: tokens[0].to_string(),
            size: tokens[1].parse::<u32>().unwrap_or(0),
        }
    } else {
        MetaTag::Unknown {
            raw: format!("TILE {}", rest),
        }
    }
}

/// `LAUNCH <bx> <by> <bz> <grid_expr>`
fn parse_launch(rest: &str) -> MetaTag {
    let tokens: Vec<&str> = rest.splitn(4, char::is_whitespace).collect();
    if tokens.len() >= 4 {
        MetaTag::Launch {
            block_x: tokens[0].parse().unwrap_or(0),
            block_y: tokens[1].parse().unwrap_or(0),
            block_z: tokens[2].parse().unwrap_or(0),
            grid_expr: tokens[3].to_string(),
        }
    } else {
        MetaTag::Unknown {
            raw: format!("LAUNCH {}", rest),
        }
    }
}

/// `SHARED_MEM <name> <elem_type> <total_bytes>`
fn parse_shared_mem(rest: &str) -> MetaTag {
    let tokens: Vec<&str> = rest.split_whitespace().collect();
    if tokens.len() >= 3 {
        MetaTag::SharedMem {
            name: tokens[0].to_string(),
            elem_type: tokens[1].to_string(),
            total_bytes: tokens[2].parse().unwrap_or(0),
        }
    } else {
        MetaTag::Unknown {
            raw: format!("SHARED_MEM {}", rest),
        }
    }
}

/// `LOOP <label> <min_iters> <max_iters> <is_unrolled>`
fn parse_loop(rest: &str) -> MetaTag {
    let tokens: Vec<&str> = rest.split_whitespace().collect();
    if tokens.len() >= 4 {
        let label = tokens[0];
        let is_unrolled = tokens[tokens.len() - 1];
        let middle = &tokens[1..tokens.len() - 1];
        let (min_str, max_str) = split_two_exprs(middle);
        MetaTag::Loop {
            label: label.to_string(),
            min_iters: eval_simple_expr(&min_str).unwrap_or(0),
            max_iters: eval_simple_expr(&max_str).unwrap_or(0),
            is_unrolled: is_unrolled == "true",
        }
    } else {
        MetaTag::Unknown {
            raw: format!("LOOP {}", rest),
        }
    }
}

/// Split a slice of tokens into two arithmetic expressions.
/// Splits at the first `number number` boundary at paren depth 0.
fn split_two_exprs(tokens: &[&str]) -> (String, String) {
    const OPS: &[&str] = &["+", "-", "*", "/"];
    let mut depth: i32 = 0;
    for i in 1..tokens.len() {
        let prev = tokens[i - 1];
        depth += prev.chars().filter(|&c| c == '(').count() as i32;
        depth -= prev.chars().filter(|&c| c == ')').count() as i32;
        if depth == 0 && !OPS.contains(&prev) && !OPS.contains(&tokens[i]) {
            return (tokens[..i].join(" "), tokens[i..].join(" "));
        }
    }
    let mid = tokens.len() / 2;
    (tokens[..mid].join(" "), tokens[mid..].join(" "))
}

/// `LAYOUT <name> <order> <dims_expr>`
fn parse_layout(rest: &str) -> MetaTag {
    let tokens: Vec<&str> = rest.splitn(3, char::is_whitespace).collect();
    if tokens.len() >= 3 {
        MetaTag::Layout {
            name: tokens[0].to_string(),
            order: tokens[1].to_string(),
            dims: tokens[2].to_string(),
        }
    } else {
        MetaTag::Unknown {
            raw: format!("LAYOUT {}", rest),
        }
    }
}

/// `CONST_TABLE <symbol_name> <num_entries>`
fn parse_const_table(rest: &str) -> MetaTag {
    let tokens: Vec<&str> = rest.split_whitespace().collect();
    if tokens.len() >= 2 {
        MetaTag::ConstTable {
            symbol_name: tokens[0].to_string(),
            num_entries: tokens[1].parse().unwrap_or(0),
        }
    } else {
        MetaTag::Unknown {
            raw: format!("CONST_TABLE {}", rest),
        }
    }
}

/// `CONST <symbol_name> <fields>`
fn parse_const(rest: &str) -> MetaTag {
    let mut parts = rest.splitn(2, char::is_whitespace);
    let symbol = parts.next().unwrap_or("").to_string();
    let fields = parts.next().unwrap_or("").trim().to_string();
    MetaTag::Const {
        symbol_name: symbol,
        fields,
    }
}

/// `CUSTOM <key> <value...>`
fn parse_custom(rest: &str) -> MetaTag {
    let mut parts = rest.splitn(2, char::is_whitespace);
    let key = parts.next().unwrap_or("").to_string();
    let value = parts.next().unwrap_or("").trim().to_string();
    MetaTag::Custom { key, value }
}

/// `VERSION <n>`
fn parse_version_tag(rest: &str) -> MetaTag {
    MetaTag::Version {
        version: rest.trim().parse().unwrap_or(0),
    }
}

fn eval_simple_expr(expr: &str) -> Option<u32> {
    meval::eval_str(expr).ok().map(|r| r as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loop_paren_expr() {
        let cases = [
            ("outer_loop 32 / 8 32 / 8 false", 4u32, 4u32, false),
            ("smem_load_loop 128 / (((128 * 128) / (8 * 8)) / 8) 128 / (((128 * 128) / (8 * 8)) / 8) false", 4, 4, false),
            ("spam_loop 1 1 false", 1, 1, false),
            ("dot_loop 8 8 false", 8, 8, false),
        ];
        for (input, exp_min, exp_max, exp_unrolled) in cases {
            let tag = parse_loop(input);
            match tag {
                MetaTag::Loop { label: _, min_iters, max_iters, is_unrolled } => {
                    assert_eq!(min_iters, exp_min, "min for '{}'", input);
                    assert_eq!(max_iters, exp_max, "max for '{}'", input);
                    assert_eq!(is_unrolled, exp_unrolled, "unrolled for '{}'", input);
                }
                other => panic!("expected Loop, got {:?} for '{}'", other, input),
            }
        }
    }
}
