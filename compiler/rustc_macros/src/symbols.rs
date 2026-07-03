//! Proc macro which builds the Symbol table
//!
//! # Debugging
//!
//! Since this proc-macro does some non-trivial work, debugging it is important.
//! This proc-macro can be invoked as an ordinary unit test, like so:
//!
//! ```bash
//! cd compiler/rustc_macros
//! cargo test symbols::test_symbols -- --nocapture
//! ```
//!
//! This unit test finds the `symbols!` invocation in `compiler/rustc_span/src/symbol.rs`
//! and runs it. It verifies that the output token stream can be parsed as valid module
//! items and that no errors were produced.
//!
//! You can also view the generated code by using `cargo expand`:
//!
//! ```bash
//! cargo install cargo-expand          # this is necessary only once
//! cd compiler/rustc_span
//! # The specific version number in CFG_RELEASE doesn't matter.
//! # The output is large.
//! CFG_RELEASE="0.0.0" cargo +nightly expand > /tmp/rustc_span.rs
//! ```

use std::collections::HashMap;

use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::parse::{Parse, ParseStream, Result};
use syn::punctuated::Punctuated;
use syn::{Expr, Ident, Lit, LitStr, Macro, Token, braced};

#[cfg(test)]
mod tests;

mod kw {
    syn::custom_keyword!(Keywords);
    syn::custom_keyword!(Symbols);
}

struct Keyword {
    name: Ident,
    value: LitStr,
}

impl Parse for Keyword {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let name = input.parse()?;
        input.parse::<Token![:]>()?;
        let value = input.parse()?;

        Ok(Keyword { name, value })
    }
}

struct Symbol {
    name: Ident,
    value: Value,
}

enum Value {
    SameAsName,
    String(LitStr),
    Env(LitStr, Macro),
    Unsupported(Expr),
}

impl Parse for Symbol {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let name = input.parse()?;
        let colon_token: Option<Token![:]> = input.parse()?;
        let value = if colon_token.is_some() { input.parse()? } else { Value::SameAsName };

        Ok(Symbol { name, value })
    }
}

impl Parse for Value {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let expr: Expr = input.parse()?;
        match &expr {
            Expr::Lit(expr) => {
                if let Lit::Str(lit) = &expr.lit {
                    return Ok(Value::String(lit.clone()));
                }
            }
            Expr::Macro(expr) => {
                if expr.mac.path.is_ident("env")
                    && let Ok(lit) = expr.mac.parse_body()
                {
                    return Ok(Value::Env(lit, expr.mac.clone()));
                }
            }
            _ => {}
        }
        Ok(Value::Unsupported(expr))
    }
}

struct Input {
    keywords: Punctuated<Keyword, Token![,]>,
    symbols: Punctuated<Symbol, Token![,]>,
}

impl Parse for Input {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        input.parse::<kw::Keywords>()?;
        let content;
        braced!(content in input);
        let keywords = Punctuated::parse_terminated(&content)?;

        input.parse::<kw::Symbols>()?;
        let content;
        braced!(content in input);
        let symbols = Punctuated::parse_terminated(&content)?;

        Ok(Input { keywords, symbols })
    }
}

#[derive(Default)]
struct Errors {
    list: Vec<syn::Error>,
}

impl Errors {
    fn error(&mut self, span: Span, message: String) {
        self.list.push(syn::Error::new(span, message));
    }
}

pub(super) fn symbols(input: TokenStream) -> TokenStream {
    let (mut output, errors) = symbols_with_errors(input);

    // If we generated any errors, then report them as compiler_error!() macro calls.
    // This lets the errors point back to the most relevant span. It also allows us
    // to report as many errors as we can during a single run.
    output.extend(errors.into_iter().map(|e| e.to_compile_error()));

    output
}

struct Predefined {
    idx: u32,
    span_of_name: Span,
}

struct Entries {
    map: HashMap<String, Predefined>,
}

impl Entries {
    fn with_capacity(capacity: usize) -> Self {
        Entries { map: HashMap::with_capacity(capacity) }
    }

    fn insert(&mut self, span: Span, s: &str, errors: &mut Errors) -> u32 {
        if let Some(prev) = self.map.get(s) {
            errors.error(span, format!("Symbol `{s}` is duplicated"));
            errors.error(prev.span_of_name, "location of previous definition".to_string());
            prev.idx
        } else {
            let idx = self.len();
            self.map.insert(s.to_string(), Predefined { idx, span_of_name: span });
            idx
        }
    }

    fn len(&self) -> u32 {
        u32::try_from(self.map.len()).expect("way too many symbols")
    }
}

// ============================================================================
// 【带读】symbols_with_errors —— `symbols!` 宏的真正主体
//
// 职责一句话:把 rustc_span/src/symbol.rs:24 处
// `symbols! { Keywords{..} Symbols{..} }` 的输入 token 流,变成三样产物
// (最后全部拼进返回的 TokenStream):
//   1. kw_generated / sym_generated 两个模块,内容是一排排
//      `pub const 名字: Symbol = Symbol::new(编号);` 常量
//      (经 symbol.rs 里 `pub use` 转发后,就是全编译器在用的 kw::/sym::);
//   2. Interner::with_extra_symbols(),内含 prefill 字符串数组——
//      数组第 idx 个字符串 == 编号为 idx 的那个符号(顺序即编号!);
//   3. 几个基址常量(SYMBOL_DIGITS_BASE 等,给单字符符号的 O(1) 快速通道用)。
//
// ★ 全函数唯一的核心不变量:每给 entries 发一个编号,必须同步往
//   prefill_stream 推入对应字符串,且顺序严格一致。因为运行期
//   Symbol::new(idx).as_str() 就是"取 prefill 数组第 idx 个"——两边一旦
//   错位,不会编译报错,而是所有符号运行期静默串号(as_str 返回别的词)。
//   下面每个循环都在守这一条,读代码时请时刻盯着它。
//
// 返回 (产物, 错误列表) 而不是 Result:出错也照常产出代码。错误由调用方
// symbols()(本文件 :134)逐条转成 compile_error!() 附在产物末尾——
// 这样报错能指向精确 span,且一次运行报出全部错误。
// ============================================================================
fn symbols_with_errors(input: TokenStream) -> (TokenStream, Vec<syn::Error>) {
    // 错误收集器(:123 定义,内部就是 Vec<syn::Error>):全函数任何地方出错
    // 都只是往这里 push,从不提前 return——"出错不停机,攒着一起报"。
    let mut errors = Errors::default();

    // 第①段:把原始 token 流解析成结构化的 Input(:102 定义)。
    // 解析逻辑分散在各 `impl Parse` 里(:48/:70/:80/:107),每个都是输入
    // 语法的一比一镜像,建议对照 symbol.rs:24 的真实输入阅读。
    let input: Input = match syn::parse2(input) {
        Ok(input) => input,
        Err(e) => {
            // 【关键设计】解析失败也 *不* 提前返回:记下错误,换一个空的
            // Input 继续走完全程。原因:若在此直接退出,宏不产出任何代码,
            // symbol.rs 里成千上万处 `sym::xxx` 会全部报"找不到名字",
            // 真正的语法错误将被海量无关错误淹没;塞空输入让(空的)模块
            // 骨架照常生成,用户只看到那一条真错——容错是为了报错信噪比。
            // 下面两行英文是上游原注释,说的正是这件事:
            // This allows us to display errors at the proper span, while minimizing
            // unrelated errors caused by bailing out (and not generating code).
            errors.list.push(e);
            Input { keywords: Default::default(), symbols: Default::default() }
        }
    };

    // 第②段:准备三条"输出流"和一个发号器。
    // 三条流都是 quote!{} 造出的空 TokenStream,后面各循环不断往里追加,
    // 最后在本函数结尾(第⑦段)的大 quote! 里各就各位:
    //   keyword_stream → kw_generated 模块体(关键字常量)
    //   symbols_stream → sym_generated 模块体(普通符号常量)
    //   prefill_stream → Interner::prefill 的字符串数组(顺序 = 编号!)
    //
    // 【名词解释:quote 为什么叫 quote】
    // 词源是 Lisp 的 quote(引用)概念:平时写代码,编译器会"执行/编译"它;
    // 而"引用"一段代码,是把它当**数据**原样保存、不执行。quote!{...} 就是
    // 干这个的:花括号里的 Rust 代码不会被编译执行,而是原样变成 TokenStream
    // 数据返回——好比字符串的引号 "..." 把文本变成数据,quote! 把代码变成
    // 数据,所以叫"引用"。其中 #变量 处例外:它"解除引用",把外部变量的值
    // 插进去(类似 format! 的 {} 占位)。这种"整体引用 + 局部插值"的写法,
    // 术语叫 quasi-quoting(准引用)。
    //
    // 【名词解释:prefill 为什么叫 prefill】
    // pre-fill = "预先填充"。Interner(字符串驻留器)本来的工作方式是运行期
    // 按需登记:每次 intern("xxx") 时查表,没有才分配新编号。而本宏在编译
    // rustc 时就已经把编号发好了(写死在 sym::xxx 常量里),所以 Interner
    // 创建的那一刻,必须把全部预定义字符串按同样的顺序一次性填进表——
    // "在使用之前预先填好",故名 prefill。这样 sym::abi 里那个编译期写死的
    // 编号,从程序第 0 秒起就能查到对应字符串。
    let mut keyword_stream = quote! {};
    let mut symbols_stream = quote! {};
    let mut prefill_stream = quote! {};
    // 除了用户列出的符号,还要静默补 "0"-"9"、"A"-"Z"、"a"-"z" 共 62 个
    // 单字符符号(见第⑤段),这两个迭代器就是它们的来源。
    let prefill_ints = 0..=9;
    let prefill_letters = ('A'..='Z').chain('a'..='z');
    // entries(:150 定义)是编号的唯一权威:字符串 → { idx, 定义位置 },
    // 兼任查重台(重复定义报双条错)。容量 = 四类符号的总数,一次分配到位。
    let mut entries = Entries::with_capacity(
        input.keywords.len()
            + input.symbols.len()
            + prefill_ints.clone().count()
            + prefill_letters.clone().count(),
    );

    // 第③段:处理 Keywords 段。每个关键字(如 `As: "as"`)三个动作一气呵成:
    // 发号 → 推字符串 → 生成常量。这是核心不变量"同一循环产出两边"的
    // 第一次执行:idx 和 #value 在相邻两行产生,想错位都难。
    // Generate the listed keywords.
    for keyword in input.keywords.iter() {
        let name = &keyword.name; // 例:As      (Ident)
        let value = &keyword.value; // 例:"as"  (LitStr,带引号的字面量 token)
        let value_string = value.value(); // 例:as (真正的字符串内容,用于查重)
        // 发号:重复则报双条错(这里重复了 + 上次定义在哪),但仍返回旧编号
        // 让代码生成继续——错误已记录,产物依然完整。
        let idx = entries.insert(keyword.name.span(), &value_string, &mut errors);
        // 同步推字符串:prefill 数组第 idx 个位置就是 #value ——不变量!
        prefill_stream.extend(quote! {
            #value,
        });
        // 生成常量:#name/#idx 是 quote! 的填空插值,
        // 产出形如 `pub const As: Symbol = Symbol::new(2);`
        keyword_stream.extend(quote! {
            pub const #name: Symbol = Symbol::new(#idx);
        });
    }

    // 第④段:处理 Symbols 段。骨架与第③段完全相同(发号→推串→生成常量),
    // 唯一区别是先要对 Value 的四个变体(:63 定义)分派出"值到底是什么"。
    // Generate the listed symbols.
    for symbol in input.symbols.iter() {
        let name = &symbol.name;

        let value = match &symbol.value {
            // 写法 `abi`         → 值就是名字本身 "abi"
            Value::SameAsName => name.to_string(),
            // 写法 `abi: "..."`  → 用显式给的字符串
            Value::String(lit) => lit.value(),
            // 写法 `x: env!(..)` → 本循环跳过,留给第⑥段单独处理
            // (因为它要读环境变量,且撞值规则和普通符号不同)
            Value::Env(..) => continue, // in another loop below
            // 其他任何表达式 → 解析期(:98)没报错、包成 Unsupported 带到这里,
            // 现在才兑现报错:new_spanned 把红线画在那个非法表达式上,
            // 错误文案里的 file!() 还顺手告诉你"想支持就来改这个文件"。
            Value::Unsupported(expr) => {
                errors.list.push(syn::Error::new_spanned(
                    expr,
                    concat!(
                        "unsupported expression for symbol value; implement support for this in ",
                        file!(),
                    ),
                ));
                continue;
            }
        };
        // 以下三步与第③段一模一样:发号、推串(守不变量)、生成常量。
        let idx = entries.insert(symbol.name.span(), &value, &mut errors);

        prefill_stream.extend(quote! {
            #value,
        });
        symbols_stream.extend(quote! {
            pub const #name: Symbol = Symbol::new(#idx);
        });
    }

    // 第⑤段:静默补 62 个单字符符号("0"-"9"、"A"-"Z"、"a"-"z")。
    // 输入里根本没写它们——这是给运行期的 O(1) 快速通道铺路:三段各自
    // 连续编号,之后驻留 "7" 这类单字符时直接算 SYMBOL_DIGITS_BASE + 7,
    // 连哈希表都不用查(见 symbol.rs 的 Symbol::integer)。
    // 注意:这里没有对应的 pub const 常量,所以只发号 + 推串两个动作。
    // Generate symbols for ascii letters and digits
    for s in prefill_ints.map(|n| n.to_string()).chain(prefill_letters.map(|c| c.to_string())) {
        entries.insert(Span::call_site(), &s, &mut errors);
        prefill_stream.extend(quote! {
            #s,
        });
    }

    // 第⑥段:处理值来自环境变量的符号(第④段跳过的 Value::Env)。
    // 上游注释第二句是本段题眼:"It's allowed for these to have the same
    // value as another symbol"——env! 符号允许和别的符号撞值(见下)。
    // 本段必须排在第⑤段之后:万一某个 env! 的值恰好是 "7" 这类单字符,
    // 需要能查到第⑤段已登记的条目并复用其编号。
    // Symbols whose value comes from an environment variable. It's allowed for
    // these to have the same value as another symbol.
    for symbol in &input.symbols {
        // 只挑 Env 变体,其余三种第④段已处理完,跳过。
        let (env_var, expr) = match &symbol.value {
            Value::Env(lit, expr) => (lit, expr),
            Value::SameAsName | Value::String(_) | Value::Unsupported(_) => continue,
        };

        // tracked::env_var 只在真正的 proc-macro 环境里可用;
        // 单元测试(cargo test)裸调本函数时没有这个环境,只能报错跳过。
        if !proc_macro::is_available() {
            errors.error(
                Span::call_site(),
                "proc_macro::tracked_env is not available in unit test".to_owned(),
            );
            break;
        }

        // 在"编译宏调用处"的这一刻读环境变量。tracked 版会把依赖登记进
        // 增量编译系统:这个环境变量变了,相关代码会被重新编译。
        let tracked_env = proc_macro::tracked::env_var(env_var.value());

        let value = match tracked_env {
            Ok(value) => value,
            Err(err) => {
                errors.list.push(syn::Error::new_spanned(expr, err));
                continue;
            }
        };

        // 【本段精华】与第③④段"撞值=报错"不同,这里撞值是合法的:
        // 普通符号的值写死在源码里,重复必是笔误;而 env! 的值随环境变化,
        // 完全可能恰好等于某个已有符号(如版本号撞上数字),这不是用户的错。
        // 处理方式:查到同值旧条目就直接复用旧编号(不推串、不发新号,
        // 不变量依然成立);查不到才走正常的 推串+发号。
        let idx = if let Some(prev) = entries.map.get(&value) {
            prev.idx
        } else {
            prefill_stream.extend(quote! {
                #value,
            });
            entries.insert(symbol.name.span(), &value, &mut errors)
        };

        // 无论编号是复用的还是新发的,常量照常生成——
        // 两个不同名字的常量可以指向同一个编号。
        let name = &symbol.name;
        symbols_stream.extend(quote! {
            pub const #name: Symbol = Symbol::new(#idx);
        });
    }

    // 第⑦段:最终组装。先取出三个基址(第⑤段登记的 "0"/"A"/"a" 的编号,
    // 三段各自连续,所以知道起点就能算出任意单字符的编号)和符号总数。
    let symbol_digits_base = entries.map["0"].idx;
    let symbol_uppercase_letters_base = entries.map["A"].idx;
    let symbol_lowercase_letters_base = entries.map["a"].idx;
    let predefined_symbols_count = entries.len();
    // 这个大 quote! 就是产物的完整模板(读宏先读这里!):
    // #keyword_stream / #symbols_stream / #prefill_stream 把一路攒下的
    // 三条流嵌进对应位置。kw_generated/sym_generated 会被 symbol.rs 里的
    // `pub use` 转发成 kw::/sym::;prefill 数组则流进 Interner::prefill——
    // 常量里的编号和驻留表里的位置出自同一次循环,不变量在此闭环。
    let output = quote! {
        const SYMBOL_DIGITS_BASE: u32 = #symbol_digits_base;
        const SYMBOL_UPPERCASE_LETTERS_BASE: u32 = #symbol_uppercase_letters_base;
        const SYMBOL_LOWERCASE_LETTERS_BASE: u32 = #symbol_lowercase_letters_base;

        /// The number of predefined symbols; this is the first index for
        /// extra pre-interned symbols in an Interner created via
        /// [`Interner::with_extra_symbols`].
        pub const PREDEFINED_SYMBOLS_COUNT: u32 = #predefined_symbols_count;

        #[doc(hidden)]
        #[allow(non_upper_case_globals)]
        mod kw_generated {
            use super::Symbol;
            #keyword_stream
        }

        #[allow(non_upper_case_globals)]
        #[doc(hidden)]
        pub mod sym_generated {
            use super::Symbol;
            #symbols_stream
        }

        impl Interner {
            /// Creates an `Interner` with the predefined symbols from the `symbols!` macro and
            /// any extra symbols provided by external drivers such as Clippy
            pub(crate) fn with_extra_symbols(extra_symbols: &[&'static str]) -> Self {
                Interner::prefill(
                    &[#prefill_stream],
                    extra_symbols,
                )
            }
        }
    };

    // 产物与错误一起交还给入口 symbols()(:134):产物原样往外传,
    // 错误在那里逐条变成 compile_error!() 追加到产物末尾。
    (output, errors.list)
}
