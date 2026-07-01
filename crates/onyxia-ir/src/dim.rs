//! Symbolic dimension algebra.
//!
//! Shapes in the IR are vectors of [`DimExpr`] — integer polynomials over
//! named dimension symbols (`sequence_length`, `batch`, …). Expressions are
//! kept in a **canonical form** (a sorted sum of terms, each a coefficient
//! times a sorted product of symbols), so structural equality *is* semantic
//! equality and simplification happens by construction.
//!
//! The algebra is deliberately small: addition, subtraction, multiplication,
//! and **exact** division by a monomial (needed to resolve ONNX `Reshape`
//! `-1` targets like `[B*S*4096] / 64`). There is no general division,
//! no min/max, no modulo — anything that escapes this fragment becomes a
//! fresh *late-bound* symbol whose value is learned at run time
//! (`doc/ir-design.md` §3).
//!
//! Coefficients are signed (`i64`) so intermediate shape arithmetic can
//! subtract (`total_len - past_len`); a dimension must evaluate to a
//! non-negative value once symbols are bound, which [`DimExpr::eval`]
//! enforces.

use crate::{Error, Result};
use std::collections::HashMap;
use std::ops::{Add, Mul, Sub};

/// Identifier of a dimension symbol, allocated by a [`SymbolTable`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SymId(u32);

impl SymId {
    /// The index of this symbol in its [`SymbolTable`].
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// Allocates and names dimension symbols.
///
/// Symbols come from ONNX `dim_param` names (`"sequence_length"`) or are
/// generated fresh for anonymous dynamic dims and late-bound fallbacks.
#[derive(Debug, Clone, Default)]
pub struct SymbolTable {
    names: Vec<String>,
    by_name: HashMap<String, SymId>,
}

impl SymbolTable {
    /// Create an empty table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the symbol with the given name, allocating it if new.
    pub fn intern(&mut self, name: &str) -> SymId {
        if let Some(&id) = self.by_name.get(name) {
            return id;
        }
        let id = SymId(self.names.len() as u32);
        self.names.push(name.to_string());
        self.by_name.insert(name.to_string(), id);
        id
    }

    /// Allocate a fresh anonymous symbol with a generated, unique name.
    pub fn fresh(&mut self, hint: &str) -> SymId {
        let mut name = format!("{hint}${}", self.names.len());
        while self.by_name.contains_key(&name) {
            name.push('\'');
        }
        self.intern(&name)
    }

    /// Look up a symbol's name.
    pub fn name(&self, id: SymId) -> &str {
        &self.names[id.index()]
    }

    /// Look up a symbol by name without allocating.
    pub fn get(&self, name: &str) -> Option<SymId> {
        self.by_name.get(name).copied()
    }

    /// Number of symbols allocated.
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// Whether no symbols have been allocated.
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

/// Concrete values for dimension symbols, established at binding time.
#[derive(Debug, Clone, Default)]
pub struct Bindings {
    values: HashMap<SymId, u64>,
}

impl Bindings {
    /// Create an empty set of bindings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Bind a symbol to a concrete value.
    ///
    /// Returns an error if the symbol is already bound to a different value —
    /// the same symbol appearing in several input shapes must be consistent.
    pub fn bind(&mut self, sym: SymId, value: u64) -> Result<()> {
        match self.values.get(&sym) {
            Some(&prev) if prev != value => Err(Error::Binding(format!(
                "symbol bound to conflicting values {prev} and {value}"
            ))),
            _ => {
                self.values.insert(sym, value);
                Ok(())
            }
        }
    }

    /// Look up a symbol's value.
    pub fn get(&self, sym: SymId) -> Option<u64> {
        self.values.get(&sym).copied()
    }
}

/// One canonical term: `coeff * syms[0] * syms[1] * …`.
///
/// `syms` is sorted and may contain repeats (`S*S`). A constant term has an
/// empty `syms`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Term {
    /// Product of symbols, sorted, with multiplicity.
    syms: Vec<SymId>,
    /// Signed coefficient, never zero in canonical form.
    coeff: i64,
}

/// A symbolic dimension: an integer polynomial over dimension symbols.
///
/// Canonical-form invariants: terms sorted by their symbol product, symbol
/// products sorted, no zero coefficients, no duplicate symbol products.
/// Structural equality is therefore semantic equality of polynomials.
///
/// ```
/// use onyxia_ir::{DimExpr, SymbolTable};
///
/// let mut syms = SymbolTable::new();
/// let s = DimExpr::sym(syms.intern("S"));
/// let four_s = s.clone() * DimExpr::constant(4);
/// assert_eq!(four_s.clone() + s.clone(), s.clone() * DimExpr::constant(5));
/// assert_eq!((four_s / &DimExpr::constant(2)).unwrap(), s * DimExpr::constant(2));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DimExpr {
    /// Sorted terms; empty means the constant 0.
    terms: Vec<Term>,
}

impl DimExpr {
    /// The constant expression `c`.
    pub fn constant(c: u64) -> Self {
        Self::from_signed(c as i64)
    }

    fn from_signed(c: i64) -> Self {
        if c == 0 {
            Self { terms: vec![] }
        } else {
            Self {
                terms: vec![Term {
                    syms: vec![],
                    coeff: c,
                }],
            }
        }
    }

    /// The expression consisting of a single symbol.
    pub fn sym(id: SymId) -> Self {
        Self {
            terms: vec![Term {
                syms: vec![id],
                coeff: 1,
            }],
        }
    }

    /// If this expression is a constant, return it.
    ///
    /// Returns `None` for constants that are negative (a negative dimension
    /// is representable mid-arithmetic but is never a valid dim on its own).
    pub fn as_const(&self) -> Option<u64> {
        match self.terms.as_slice() {
            [] => Some(0),
            [t] if t.syms.is_empty() => u64::try_from(t.coeff).ok(),
            _ => None,
        }
    }

    /// Whether this expression contains no symbols.
    pub fn is_const(&self) -> bool {
        self.terms.iter().all(|t| t.syms.is_empty())
    }

    /// If this expression is exactly one bare symbol (`S`), return it.
    pub fn as_sym(&self) -> Option<SymId> {
        match self.terms.as_slice() {
            [t] if t.coeff == 1 && t.syms.len() == 1 => Some(t.syms[0]),
            _ => None,
        }
    }

    /// All symbols referenced by this expression (with duplicates removed).
    pub fn syms(&self) -> Vec<SymId> {
        let mut out: Vec<SymId> = self.terms.iter().flat_map(|t| t.syms.clone()).collect();
        out.sort();
        out.dedup();
        out
    }

    /// Evaluate under `bindings`.
    ///
    /// Errors if a symbol is unbound or the result is negative.
    pub fn eval(&self, bindings: &Bindings) -> Result<u64> {
        let mut acc: i128 = 0;
        for term in &self.terms {
            let mut prod: i128 = term.coeff as i128;
            for &sym in &term.syms {
                let v = bindings.get(sym).ok_or_else(|| {
                    Error::Binding(format!("unbound dimension symbol (id {})", sym.0))
                })?;
                prod *= v as i128;
            }
            acc += prod;
        }
        u64::try_from(acc)
            .map_err(|_| Error::Binding(format!("dimension evaluated to negative value {acc}")))
    }

    /// Exact division by a *monomial* (a single-term expression).
    ///
    /// Returns `None` if the divisor is not a monomial, or if any term's
    /// coefficient or symbol product is not exactly divisible. This is the
    /// fragment needed to resolve `Reshape` inferred (`-1`) dimensions.
    pub fn div_exact(&self, divisor: &DimExpr) -> Option<DimExpr> {
        let [d] = divisor.terms.as_slice() else {
            return None; // not a monomial (0 or a sum)
        };
        let mut terms = Vec::with_capacity(self.terms.len());
        for t in &self.terms {
            if t.coeff % d.coeff != 0 {
                return None;
            }
            // Remove the divisor's symbol multiset from the term's.
            let mut remaining = t.syms.clone();
            for dsym in &d.syms {
                let pos = remaining.iter().position(|s| s == dsym)?;
                remaining.remove(pos);
            }
            terms.push(Term {
                syms: remaining,
                coeff: t.coeff / d.coeff,
            });
        }
        Some(Self::canonicalize(terms))
    }

    /// Render with symbol names from `table`, e.g. `"4*S + 8"`.
    pub fn display<'a>(&'a self, table: &'a SymbolTable) -> impl std::fmt::Display + 'a {
        DimDisplay {
            expr: self,
            table: Some(table),
        }
    }

    fn canonicalize(mut terms: Vec<Term>) -> Self {
        for t in &mut terms {
            t.syms.sort();
        }
        // Higher-degree terms first (so `4*S*S - 2*S + 8` reads naturally);
        // any total order works for canonicalization, this one also reads
        // well in Display.
        terms.sort_by(|a, b| b.syms.cmp(&a.syms));
        let mut out: Vec<Term> = Vec::with_capacity(terms.len());
        for t in terms {
            match out.last_mut() {
                Some(last) if last.syms == t.syms => last.coeff += t.coeff,
                _ => out.push(t),
            }
        }
        out.retain(|t| t.coeff != 0);
        Self { terms: out }
    }
}

impl Add for DimExpr {
    type Output = DimExpr;
    fn add(self, rhs: DimExpr) -> DimExpr {
        let mut terms = self.terms;
        terms.extend(rhs.terms);
        DimExpr::canonicalize(terms)
    }
}

impl Sub for DimExpr {
    type Output = DimExpr;
    fn sub(self, rhs: DimExpr) -> DimExpr {
        let mut terms = self.terms;
        terms.extend(rhs.terms.into_iter().map(|mut t| {
            t.coeff = -t.coeff;
            t
        }));
        DimExpr::canonicalize(terms)
    }
}

impl Mul for DimExpr {
    type Output = DimExpr;
    fn mul(self, rhs: DimExpr) -> DimExpr {
        let mut terms = Vec::with_capacity(self.terms.len() * rhs.terms.len());
        for a in &self.terms {
            for b in &rhs.terms {
                let mut syms = a.syms.clone();
                syms.extend_from_slice(&b.syms);
                terms.push(Term {
                    syms,
                    coeff: a.coeff * b.coeff,
                });
            }
        }
        DimExpr::canonicalize(terms)
    }
}

/// `expr / &divisor` — sugar for [`DimExpr::div_exact`].
impl std::ops::Div<&DimExpr> for DimExpr {
    type Output = Option<DimExpr>;
    fn div(self, rhs: &DimExpr) -> Option<DimExpr> {
        self.div_exact(rhs)
    }
}

impl From<u64> for DimExpr {
    fn from(c: u64) -> Self {
        DimExpr::constant(c)
    }
}

struct DimDisplay<'a> {
    expr: &'a DimExpr,
    table: Option<&'a SymbolTable>,
}

impl std::fmt::Display for DimDisplay<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.expr.terms.is_empty() {
            return f.write_str("0");
        }
        for (i, t) in self.expr.terms.iter().enumerate() {
            if i > 0 {
                f.write_str(if t.coeff < 0 { " - " } else { " + " })?;
            } else if t.coeff < 0 {
                f.write_str("-")?;
            }
            let mag = t.coeff.unsigned_abs();
            if t.syms.is_empty() {
                write!(f, "{mag}")?;
            } else {
                if mag != 1 {
                    write!(f, "{mag}*")?;
                }
                for (j, s) in t.syms.iter().enumerate() {
                    if j > 0 {
                        f.write_str("*")?;
                    }
                    match self.table {
                        Some(tab) => f.write_str(tab.name(*s))?,
                        None => write!(f, "s{}", s.0)?,
                    }
                }
            }
        }
        Ok(())
    }
}

/// Without a symbol table, symbols render by id as `s0`, `s1`, ….
impl std::fmt::Display for DimExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        DimDisplay {
            expr: self,
            table: None,
        }
        .fmt(f)
    }
}

/// A tensor shape: one [`DimExpr`] per dimension. Rank is always known.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct SymbolicShape(pub Vec<DimExpr>);

impl SymbolicShape {
    /// A fully static shape.
    pub fn fixed(dims: &[u64]) -> Self {
        Self(dims.iter().map(|&d| DimExpr::constant(d)).collect())
    }

    /// Number of dimensions.
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// The dimensions.
    pub fn dims(&self) -> &[DimExpr] {
        &self.0
    }

    /// Total element count as a symbolic expression.
    pub fn numel(&self) -> DimExpr {
        self.0
            .iter()
            .fold(DimExpr::constant(1), |acc, d| acc * d.clone())
    }

    /// If fully static, the concrete dimensions.
    pub fn as_static(&self) -> Option<Vec<u64>> {
        self.0.iter().map(|d| d.as_const()).collect()
    }

    /// Whether every dimension is a constant.
    pub fn is_static(&self) -> bool {
        self.0.iter().all(|d| d.is_const())
    }

    /// Evaluate to concrete dimensions under `bindings`.
    pub fn eval(&self, bindings: &Bindings) -> Result<Vec<usize>> {
        self.0
            .iter()
            .map(|d| d.eval(bindings).map(|v| v as usize))
            .collect()
    }
}

impl std::fmt::Display for SymbolicShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("[")?;
        for (i, d) in self.0.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{d}")?;
        }
        f.write_str("]")
    }
}

impl From<Vec<DimExpr>> for SymbolicShape {
    fn from(dims: Vec<DimExpr>) -> Self {
        Self(dims)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(table: &mut SymbolTable, name: &str) -> DimExpr {
        DimExpr::sym(table.intern(name))
    }

    #[test]
    fn algebra_identities() {
        let mut t = SymbolTable::new();
        let x = s(&mut t, "S");

        // (S*1) + 0 == S
        assert_eq!(x.clone() * DimExpr::constant(1) + DimExpr::constant(0), x);
        // S - S == 0
        assert_eq!(x.clone() - x.clone(), DimExpr::constant(0));
        // S + S == 2*S
        assert_eq!(x.clone() + x.clone(), x.clone() * DimExpr::constant(2));
        // (S+2)*(S+3) == S*S + 5*S + 6
        let lhs = (x.clone() + DimExpr::constant(2)) * (x.clone() + DimExpr::constant(3));
        let rhs = x.clone() * x.clone() + x.clone() * DimExpr::constant(5) + DimExpr::constant(6);
        assert_eq!(lhs, rhs);
        // Multiplication commutes.
        let y = s(&mut t, "T");
        assert_eq!(x.clone() * y.clone(), y.clone() * x.clone());
    }

    #[test]
    fn eval_round_trip() {
        let mut t = SymbolTable::new();
        let sid = t.intern("S");
        let tid = t.intern("T");
        // 4*S*T + 2*S + 7
        let e = DimExpr::constant(4) * DimExpr::sym(sid) * DimExpr::sym(tid)
            + DimExpr::constant(2) * DimExpr::sym(sid)
            + DimExpr::constant(7);
        let mut b = Bindings::new();
        b.bind(sid, 3).unwrap();
        b.bind(tid, 5).unwrap();
        assert_eq!(e.eval(&b).unwrap(), 4 * 3 * 5 + 2 * 3 + 7);
    }

    #[test]
    fn eval_errors() {
        let mut t = SymbolTable::new();
        let sid = t.intern("S");
        let e = DimExpr::sym(sid) - DimExpr::constant(10);
        let mut b = Bindings::new();
        b.bind(sid, 3).unwrap();
        // 3 - 10 is negative.
        assert!(e.eval(&b).is_err());
        // Unbound symbol.
        let unbound = DimExpr::sym(t.intern("U"));
        assert!(unbound.eval(&b).is_err());
        // Conflicting rebind.
        assert!(b.bind(sid, 4).is_err());
        // Consistent rebind is fine.
        assert!(b.bind(sid, 3).is_ok());
    }

    #[test]
    fn div_exact() {
        let mut t = SymbolTable::new();
        let x = s(&mut t, "S");
        let y = s(&mut t, "T");

        // (4*S*T + 2*S) / 2 == 2*S*T + S
        let e = DimExpr::constant(4) * x.clone() * y.clone() + DimExpr::constant(2) * x.clone();
        let expect = DimExpr::constant(2) * x.clone() * y.clone() + x.clone();
        assert_eq!(e.clone().div_exact(&DimExpr::constant(2)), Some(expect));

        // (4*S*T + 2*S) / S == 4*T + 2
        let expect = DimExpr::constant(4) * y.clone() + DimExpr::constant(2);
        assert_eq!(e.clone().div_exact(&x), Some(expect));

        // Not divisible: by 3, or by T (the 2*S term has no T).
        assert_eq!(e.clone().div_exact(&DimExpr::constant(3)), None);
        assert_eq!(e.clone().div_exact(&y), None);
        // Divisor must be a monomial.
        assert_eq!(e.clone().div_exact(&(x.clone() + y.clone())), None);
        // Division by zero is not a thing.
        assert_eq!(e.div_exact(&DimExpr::constant(0)), None);

        // The Reshape(-1) case: (B*S*4096) / 64 == B*S*64
        let b = s(&mut t, "B");
        let numel = b.clone() * x.clone() * DimExpr::constant(4096);
        let inferred = numel.div_exact(&DimExpr::constant(64)).unwrap();
        assert_eq!(inferred, b * x * DimExpr::constant(64));
    }

    #[test]
    fn as_const_and_negative() {
        assert_eq!(DimExpr::constant(0).as_const(), Some(0));
        assert_eq!(DimExpr::constant(42).as_const(), Some(42));
        let neg = DimExpr::constant(1) - DimExpr::constant(3);
        assert!(neg.is_const());
        assert_eq!(neg.as_const(), None); // negative constants are not dims
    }

    #[test]
    fn display() {
        let mut t = SymbolTable::new();
        let x = s(&mut t, "S");
        let e = DimExpr::constant(4) * x.clone() * x.clone() + DimExpr::constant(8)
            - DimExpr::constant(2) * x.clone();
        assert_eq!(e.display(&t).to_string(), "4*S*S - 2*S + 8");
        assert_eq!(DimExpr::constant(0).display(&t).to_string(), "0");
        // Without a table, symbols render by id.
        assert_eq!(x.to_string(), "s0");
    }

    #[test]
    fn shape_helpers() {
        let mut t = SymbolTable::new();
        let sid = t.intern("S");
        let shape = SymbolicShape(vec![
            DimExpr::constant(1),
            DimExpr::sym(sid),
            DimExpr::constant(256),
        ]);
        assert_eq!(shape.rank(), 3);
        assert!(!shape.is_static());
        assert_eq!(shape.as_static(), None);
        assert_eq!(shape.numel(), DimExpr::sym(sid) * DimExpr::constant(256));

        let mut b = Bindings::new();
        b.bind(sid, 7).unwrap();
        assert_eq!(shape.eval(&b).unwrap(), vec![1, 7, 256]);

        let fixed = SymbolicShape::fixed(&[2, 3]);
        assert_eq!(fixed.as_static(), Some(vec![2, 3]));
        assert_eq!(fixed.to_string(), "[2, 3]");
    }

    #[test]
    fn symbol_table() {
        let mut t = SymbolTable::new();
        let a = t.intern("batch");
        let b = t.intern("batch");
        assert_eq!(a, b);
        let f1 = t.fresh("late");
        let f2 = t.fresh("late");
        assert_ne!(f1, f2);
        assert_eq!(t.name(a), "batch");
        assert_eq!(t.get("batch"), Some(a));
        assert_eq!(t.get("nope"), None);
    }
}
