//! Symbolic dimension expression parser and evaluator.
//!
//! Supports arithmetic expressions with variables to enable dimension computation
//! from base dimension values. While ONNX spec says dimension parameters "SHOULD
//! adhere to C90 identifier syntax rules", real-world models (esp. from PyTorch)
//! contain expressions like "sequence_length * num_attention_heads".
//!
//! Supported operations: `+`, `-`, `*`, `/`, `%`, parentheses
//!
//! # Examples
//!
//! ```
//! use onyxia_compiler::symbolic_expr::{parse_expr, evaluate_expr};
//! use std::collections::HashMap;
//!
//! // Simple variable reference
//! let expr = parse_expr("batch_size").unwrap();
//! let vars = HashMap::from([("batch_size".to_string(), 1)]);
//! assert_eq!(evaluate_expr(&expr, &vars).unwrap(), 1);
//!
//! // Arithmetic expression
//! let expr = parse_expr("seq * heads").unwrap();
//! let vars = HashMap::from([("seq".to_string(), 64), ("heads".to_string(), 8)]);
//! assert_eq!(evaluate_expr(&expr, &vars).unwrap(), 512);
//!
//! // Complex expression with parentheses
//! let expr = parse_expr("(batch + 1) * seq").unwrap();
//! let vars = HashMap::from([("batch".to_string(), 2), ("seq".to_string(), 128)]);
//! assert_eq!(evaluate_expr(&expr, &vars).unwrap(), 384);
//! ```

use std::collections::HashMap;

/// Abstract syntax tree for arithmetic expressions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {
    /// Integer literal
    Literal(i64),
    /// Variable reference (dimension name)
    Variable(String),
    /// Binary operation
    BinOp(Box<Expr>, BinOpKind, Box<Expr>),
}

/// Binary operator kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOpKind {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
}

/// Parse an arithmetic expression string into an AST.
///
/// # Grammar
///
/// ```text
/// expr   = term (('+' | '-') term)*
/// term   = factor (('*' | '/' | '%') factor)*
/// factor = '(' expr ')' | number | ident
/// number = [0-9]+
/// ident  = [a-zA-Z_][a-zA-Z0-9_]*
/// ```
pub fn parse_expr(input: &str) -> Result<Expr, String> {
    let tokens = tokenize(input)?;
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expr()?;

    // Ensure all tokens were consumed
    if parser.pos < parser.tokens.len() {
        return Err(format!(
            "Unexpected token after expression: {:?}",
            parser.tokens[parser.pos]
        ));
    }

    Ok(expr)
}

/// Evaluate an expression AST with given variable values.
///
/// Returns the computed integer value or an error if:
/// - A variable is not found in the provided map
/// - Division by zero occurs
/// - Integer overflow occurs
pub fn evaluate_expr(expr: &Expr, vars: &HashMap<String, usize>) -> Result<usize, String> {
    match expr {
        Expr::Literal(n) => {
            if *n < 0 {
                return Err(format!("Dimension cannot be negative: {}", n));
            }
            Ok(*n as usize)
        }
        Expr::Variable(name) => vars
            .get(name)
            .copied()
            .ok_or_else(|| format!("Variable '{}' not found in dynamic_dimensions", name)),
        Expr::BinOp(left, op, right) => {
            let left_val = evaluate_expr(left, vars)?;
            let right_val = evaluate_expr(right, vars)?;

            let result = match op {
                BinOpKind::Add => left_val
                    .checked_add(right_val)
                    .ok_or_else(|| format!("Integer overflow: {} + {}", left_val, right_val))?,
                BinOpKind::Sub => left_val
                    .checked_sub(right_val)
                    .ok_or_else(|| format!("Integer underflow: {} - {}", left_val, right_val))?,
                BinOpKind::Mul => left_val
                    .checked_mul(right_val)
                    .ok_or_else(|| format!("Integer overflow: {} * {}", left_val, right_val))?,
                BinOpKind::Div => {
                    if right_val == 0 {
                        return Err(format!("Division by zero: {} / {}", left_val, right_val));
                    }
                    left_val / right_val
                }
                BinOpKind::Mod => {
                    if right_val == 0 {
                        return Err(format!("Modulo by zero: {} % {}", left_val, right_val));
                    }
                    left_val % right_val
                }
            };

            Ok(result)
        }
    }
}

// ============================================================================
// Lexer
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
    Number(i64),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    LParen,
    RParen,
}

fn tokenize(input: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let mut chars = input.chars().peekable();

    while let Some(&ch) = chars.peek() {
        match ch {
            ' ' | '\t' | '\n' | '\r' => {
                chars.next();
            }
            '+' => {
                tokens.push(Token::Plus);
                chars.next();
            }
            '-' => {
                tokens.push(Token::Minus);
                chars.next();
            }
            '*' => {
                tokens.push(Token::Star);
                chars.next();
            }
            '/' => {
                tokens.push(Token::Slash);
                chars.next();
            }
            '%' => {
                tokens.push(Token::Percent);
                chars.next();
            }
            '(' => {
                tokens.push(Token::LParen);
                chars.next();
            }
            ')' => {
                tokens.push(Token::RParen);
                chars.next();
            }
            '0'..='9' => {
                let num = parse_number(&mut chars)?;
                tokens.push(Token::Number(num));
            }
            'a'..='z' | 'A'..='Z' | '_' => {
                let ident = parse_ident(&mut chars);
                tokens.push(Token::Ident(ident));
            }
            _ => {
                return Err(format!("Unexpected character: '{}'", ch));
            }
        }
    }

    Ok(tokens)
}

fn parse_number(chars: &mut std::iter::Peekable<std::str::Chars>) -> Result<i64, String> {
    let mut num_str = String::new();
    while let Some(&ch) = chars.peek() {
        if ch.is_ascii_digit() {
            num_str.push(ch);
            chars.next();
        } else {
            break;
        }
    }
    num_str
        .parse()
        .map_err(|_| format!("Failed to parse number: {}", num_str))
}

fn parse_ident(chars: &mut std::iter::Peekable<std::str::Chars>) -> String {
    let mut ident = String::new();
    while let Some(&ch) = chars.peek() {
        if ch.is_alphanumeric() || ch == '_' {
            ident.push(ch);
            chars.next();
        } else {
            break;
        }
    }
    ident
}

// ============================================================================
// Parser
// ============================================================================

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn current(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) {
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
    }

    fn parse_expr(&mut self) -> Result<Expr, String> {
        self.parse_add_sub()
    }

    fn parse_add_sub(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_mul_div_mod()?;

        while let Some(token) = self.current() {
            let op = match token {
                Token::Plus => BinOpKind::Add,
                Token::Minus => BinOpKind::Sub,
                _ => break,
            };

            self.advance();
            let right = self.parse_mul_div_mod()?;
            left = Expr::BinOp(Box::new(left), op, Box::new(right));
        }

        Ok(left)
    }

    fn parse_mul_div_mod(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_factor()?;

        while let Some(token) = self.current() {
            let op = match token {
                Token::Star => BinOpKind::Mul,
                Token::Slash => BinOpKind::Div,
                Token::Percent => BinOpKind::Mod,
                _ => break,
            };

            self.advance();
            let right = self.parse_factor()?;
            left = Expr::BinOp(Box::new(left), op, Box::new(right));
        }

        Ok(left)
    }

    fn parse_factor(&mut self) -> Result<Expr, String> {
        match self.current() {
            Some(Token::Number(n)) => {
                let n = *n;
                self.advance();
                Ok(Expr::Literal(n))
            }
            Some(Token::Ident(name)) => {
                let name = name.clone();
                self.advance();
                Ok(Expr::Variable(name))
            }
            Some(Token::LParen) => {
                self.advance();
                let expr = self.parse_expr()?;
                match self.current() {
                    Some(Token::RParen) => {
                        self.advance();
                        Ok(expr)
                    }
                    _ => Err("Expected closing parenthesis ')'".to_string()),
                }
            }
            _ => Err("Expected number, identifier, or '('".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_identifier() {
        let expr = parse_expr("batch_size").unwrap();
        assert_eq!(expr, Expr::Variable("batch_size".to_string()));
    }

    #[test]
    fn test_parse_number() {
        let expr = parse_expr("42").unwrap();
        assert_eq!(expr, Expr::Literal(42));
    }

    #[test]
    fn test_parse_addition() {
        let expr = parse_expr("a + b").unwrap();
        assert!(matches!(expr, Expr::BinOp(_, BinOpKind::Add, _)));
    }

    #[test]
    fn test_parse_multiplication() {
        let expr = parse_expr("seq * heads").unwrap();
        assert!(matches!(expr, Expr::BinOp(_, BinOpKind::Mul, _)));
    }

    #[test]
    fn test_parse_complex_expression() {
        let expr = parse_expr("(batch + 1) * seq / 2").unwrap();
        // Should parse correctly with proper precedence
        assert!(matches!(expr, Expr::BinOp(_, BinOpKind::Div, _)));
    }

    #[test]
    fn test_parse_operator_precedence() {
        let expr = parse_expr("a + b * c").unwrap();
        // Should be: a + (b * c), not (a + b) * c
        match expr {
            Expr::BinOp(_, BinOpKind::Add, right) => {
                assert!(matches!(*right, Expr::BinOp(_, BinOpKind::Mul, _)));
            }
            _ => panic!("Expected addition at top level"),
        }
    }

    #[test]
    fn test_evaluate_simple_variable() {
        let expr = parse_expr("batch_size").unwrap();
        let vars = HashMap::from([("batch_size".to_string(), 1)]);
        assert_eq!(evaluate_expr(&expr, &vars).unwrap(), 1);
    }

    #[test]
    fn test_evaluate_literal() {
        let expr = parse_expr("42").unwrap();
        let vars = HashMap::new();
        assert_eq!(evaluate_expr(&expr, &vars).unwrap(), 42);
    }

    #[test]
    fn test_evaluate_multiplication() {
        let expr = parse_expr("seq * heads").unwrap();
        let vars = HashMap::from([("seq".to_string(), 64), ("heads".to_string(), 8)]);
        assert_eq!(evaluate_expr(&expr, &vars).unwrap(), 512);
    }

    #[test]
    fn test_evaluate_complex_expression() {
        let expr = parse_expr("(batch + 1) * seq / 2").unwrap();
        let vars = HashMap::from([("batch".to_string(), 2), ("seq".to_string(), 128)]);
        assert_eq!(evaluate_expr(&expr, &vars).unwrap(), 192); // (2 + 1) * 128 / 2 = 192
    }

    #[test]
    fn test_evaluate_gemma_expression() {
        // Real expression from Gemma model
        let expr = parse_expr("sequence_length * num_attention_heads").unwrap();
        let vars = HashMap::from([
            ("sequence_length".to_string(), 64),
            ("num_attention_heads".to_string(), 8),
        ]);
        assert_eq!(evaluate_expr(&expr, &vars).unwrap(), 512);
    }

    #[test]
    fn test_evaluate_missing_variable() {
        let expr = parse_expr("a * b").unwrap();
        let vars = HashMap::from([("a".to_string(), 10)]);
        let err = evaluate_expr(&expr, &vars).unwrap_err();
        assert!(err.contains("Variable 'b' not found"));
    }

    #[test]
    fn test_evaluate_division_by_zero() {
        let expr = parse_expr("a / 0").unwrap();
        let vars = HashMap::from([("a".to_string(), 10)]);
        let err = evaluate_expr(&expr, &vars).unwrap_err();
        assert!(err.contains("Division by zero"));
    }

    #[test]
    fn test_evaluate_modulo_by_zero() {
        let expr = parse_expr("a % 0").unwrap();
        let vars = HashMap::from([("a".to_string(), 10)]);
        let err = evaluate_expr(&expr, &vars).unwrap_err();
        assert!(err.contains("Modulo by zero"));
    }

    #[test]
    fn test_parse_whitespace() {
        let expr1 = parse_expr("a+b").unwrap();
        let expr2 = parse_expr(" a + b ").unwrap();
        let expr3 = parse_expr("a  +  b").unwrap();
        assert_eq!(expr1, expr2);
        assert_eq!(expr2, expr3);
    }

    #[test]
    fn test_parse_parentheses() {
        let expr = parse_expr("(a + b) * (c + d)").unwrap();
        assert!(matches!(expr, Expr::BinOp(_, BinOpKind::Mul, _)));
    }

    #[test]
    fn test_invalid_syntax() {
        assert!(parse_expr("a +").is_err());
        assert!(parse_expr("* b").is_err());
        assert!(parse_expr("(a + b").is_err());
        assert!(parse_expr("a b").is_err());
    }

    #[test]
    fn test_all_operators() {
        let vars = HashMap::from([("a".to_string(), 10), ("b".to_string(), 3)]);

        assert_eq!(
            evaluate_expr(&parse_expr("a + b").unwrap(), &vars).unwrap(),
            13
        );
        assert_eq!(
            evaluate_expr(&parse_expr("a - b").unwrap(), &vars).unwrap(),
            7
        );
        assert_eq!(
            evaluate_expr(&parse_expr("a * b").unwrap(), &vars).unwrap(),
            30
        );
        assert_eq!(
            evaluate_expr(&parse_expr("a / b").unwrap(), &vars).unwrap(),
            3
        );
        assert_eq!(
            evaluate_expr(&parse_expr("a % b").unwrap(), &vars).unwrap(),
            1
        );
    }

    #[test]
    fn test_underscores_in_identifiers() {
        let expr = parse_expr("sequence_length * num_attention_heads").unwrap();
        let vars = HashMap::from([
            ("sequence_length".to_string(), 64),
            ("num_attention_heads".to_string(), 8),
        ]);
        assert_eq!(evaluate_expr(&expr, &vars).unwrap(), 512);
    }

    #[test]
    fn test_literal_arithmetic() {
        let vars = HashMap::new();
        assert_eq!(
            evaluate_expr(&parse_expr("2 + 3").unwrap(), &vars).unwrap(),
            5
        );
        assert_eq!(
            evaluate_expr(&parse_expr("10 - 4").unwrap(), &vars).unwrap(),
            6
        );
        assert_eq!(
            evaluate_expr(&parse_expr("5 * 6").unwrap(), &vars).unwrap(),
            30
        );
        assert_eq!(
            evaluate_expr(&parse_expr("20 / 4").unwrap(), &vars).unwrap(),
            5
        );
    }
}
