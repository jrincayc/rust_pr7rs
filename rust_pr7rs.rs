/* Copyright 2021 Joshua J. Cogliati

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

extern crate core;
use core::fmt;
use std::io;
use std::io::Write;
use std::collections::HashMap;
use std::rc::Rc;


enum Token{
    IntegerToken(i64),
    StringToken(String),
    TokenList(Vec<Token>),
    QuoteToken(Box<Token>),
    Dot,
}

impl<'a> fmt::Debug for Token {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> fmt::Result
    {
        let mut result = Ok(());
        match self {
            &Token::IntegerToken(value) =>
                result = result.and(write!(f, "{}", value)),
            &Token::StringToken(ref value) =>
                result = result.and(write!(f, "{}", value)),
            &Token::TokenList(ref list) =>
                result = result.and(write!(f, "{:?}", list)),
            &Token::QuoteToken(ref value) =>
                result = result.and(write!(f, "'{:?}", value)),
            &Token::Dot =>
                result = result.and(write!(f, " . "))
        }
        result
    }
}

impl Token {
    fn str(s: &str) -> Token
    {
        Token::StringToken(String::from(s))
    }
}

impl Clone for Token {
    fn clone(&self) -> Token {
        match self {
            &Token::IntegerToken(value) => Token::IntegerToken(value),
            &Token::StringToken(ref value) => Token::StringToken(value.clone()),
            &Token::TokenList(ref list) => Token::TokenList(list.clone()),
            &Token::QuoteToken(ref value) => Token::QuoteToken(value.clone()),
            &Token::Dot => Token::Dot
        }
    }
}

fn tokenize(chars: &str) -> Vec<&str> {
    let mut tokens = vec![];
    let mut buffer_start = 0;
    let mut buffer_len: usize = 0;
    let mut in_comment = false;
    fn switch_token<'a>(ctokens:&mut Vec<&'a str>,cchars:&'a str,
                buffer_start:usize,buffer_len:&mut usize){
        if *buffer_len > 0
        { (*ctokens).push(&cchars[buffer_start..
                                   buffer_start+*buffer_len]);
            *buffer_len = 0;}
    }

    for (ind,l) in chars.char_indices() {
        match l {
            '\n' => {
                switch_token(&mut tokens,chars,buffer_start,&mut buffer_len);
                in_comment = false;
            },
            _ if in_comment => buffer_len = 0,
            ' ' | '\t'  =>
                switch_token(&mut tokens,chars,buffer_start,&mut buffer_len),
            '(' =>
              {switch_token(&mut tokens,chars,buffer_start,&mut buffer_len);
               tokens.push("(")
              },
              ')' =>
            {switch_token(&mut tokens,chars,buffer_start,&mut buffer_len);
             tokens.push(")")},
            '\'' =>
                tokens.push("\'"),
            ';' => in_comment = true,
            _ => {if buffer_len == 0
            { buffer_start = ind }
            buffer_len += 1}
        }
    }
    switch_token(&mut tokens,chars, buffer_start,&mut buffer_len);
    return tokens;
}

fn atom(token: &str) -> Token {
    let int_result = token.parse::<i64>();
    match int_result {
        Ok(value) => return Token::IntegerToken(value),
        _ => ()
    };
    if token == "." {
        return Token::Dot
    }
    Token::str(token)
}

enum ParseOption {
    None,
    Partial,
    Some(Token),
}

fn read_from_tokens<'a>(tokens: &Vec<&'a str>, index: &mut usize) -> ParseOption {
    if tokens.len() == 0 {
        println!("Unexpected end of program");
        return ParseOption::None
    };
    //let mut index: usize = 0;
    let first = tokens[*index];
    //println!("first {}",first);
    if first == "(" {
        let mut token_list = vec![];
        loop {
            *index += 1;
            if *index >= tokens.len() {
                //No closing ), so probably need to read more input
                return ParseOption::Partial
            }
            if tokens[*index] == ")" {break};
            let result = read_from_tokens(tokens,index);
            //println!("result {}",result);
            match result {
                ParseOption::Some(value) =>
                    token_list.push(value),
                ParseOption::None => {println!("Failed above");
                                      return ParseOption::None;},
                ParseOption::Partial => {
                    return ParseOption::Partial;
                }
            }
        }
        //*index += 1; //Remove ")"
        return ParseOption::Some(Token::TokenList(token_list));
    } else if first == "'" {
        *index += 1;
        let result = read_from_tokens(tokens, index);
        match result {
            ParseOption::Some(value) => {
                return ParseOption::Some(Token::QuoteToken(Box::new(value)));
            }
            ParseOption::None => {println!("Failed after quote");
                     return ParseOption::None;
            }
            ParseOption::Partial => {
                return ParseOption::Partial;
            }
        }
    } else if first == ")" {
        println!("Syntax error unexpected )");
        return ParseOption::None; }
    else {
        return ParseOption::Some(atom(first));
    }
}

fn parse(program: &str) ->  ParseOption {
    let mut index = 0;
    read_from_tokens(&tokenize(program), &mut index)
}

enum Value {
    Integer(i64),
    Boolean(bool),
    Symbol(String),
    Pair(Rc<Value>, Rc<Value>),
    Function(fn(Vec<Rc<Value>>) -> Rc<Value>),
    //RefFunction(fn(Vec<Value>) -> Value),
    Undefined,
    EmptyList,
    Procedure(Procedure),
    Partial(Rc<Token>, Rc<Env>), //Needs to be evaluated
    Error(String)
}

impl<'a> fmt::Debug for Value {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> fmt::Result
    {
        let mut result = Ok(());
        match self {
            &Value::Integer(value) =>
                result = result.and(write!(f, "{}", value)),
            &Value::Boolean(true) =>
                result = result.and(write!(f, "#t")),
            &Value::Boolean(false) =>
                result = result.and(write!(f, "#f")),
            &Value::Symbol(ref value) =>
                result = result.and(write!(f, "{}", value)),
            &Value::Pair(ref car, ref cdr) =>
            {
                result = result.and(write!(f, "({:?}", car));
                let mut cur_cdr = cdr;
                let mut done = false;
                while !done {
                    if let Value::Pair(ref cadr, ref cddr) = **cur_cdr {
                        result = result.and(write!(f, " {:?}", cadr));
                        cur_cdr = cddr
                    }
                    else if let Value::EmptyList = **cur_cdr {
                        result = result.and(write!(f, ")"));
                        done = true
                    } else {
                        result = result.and(write!(f, " . {:?})", cur_cdr));
                        done = true
                    }
                }
            },
            &Value::Function(_) =>
                result = result.and(write!(f, "<function>")),
            &Value::Procedure(_) =>
                result = result.and(write!(f, "<procedure>")),
            /* &Value::RefFunction(_) =>
            result = result.and(write!(f, "show not implemented for ref function")),*/
            &Value::EmptyList => result = result.and(write!(f, "()")),
            &Value::Undefined =>
                result = result.and(write!(f, "Undefined")),
            &Value::Partial(_, _) =>
                result = result.and(write!(f, "<partial>")),
            &Value::Error(ref value) =>
                result = result.and(write!(f, "Error({})", value))
        }
        result
    }
}


type REnv = Rc<Env>;
type PartialEnv = HashMap<String, Rc<Value>>;

struct Env {
    inner: PartialEnv,
    outer_env: Option<REnv>,
}


impl<'a> fmt::Debug for Env {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> fmt::Result
    {
        let mut result = Ok(());
        result = result.and(write!(f, "inner: {:?} outer: {:?}", self.inner, self.outer_env));
        result
    }
}

impl Env {
    fn new(hashmap: PartialEnv) -> Env {
        Env {
            inner: hashmap,
            outer_env: None
        }
    }
    fn new_sub(hashmap: PartialEnv, outer: Rc<Env>) -> Env {
        Env {
            inner: hashmap,
            outer_env : Some(outer)
        }
    }

    fn find(&self, key: &String) -> Option<Rc<Value>> {
        let in_current = self.inner.get(key);
        match in_current {
            Some(_) => in_current.cloned(),
            None => {
                match self.outer_env {
                    Some(ref outer) => outer.find(key),
                    None => None
                }
            }
        }
    }
}


struct Procedure {
    params: Vec<String>,
    fixed_num: bool, //If true put all parameters list and put in one var
    body: Rc<Token>,
    declarations: Vec<Rc<Token>>,
    outer_env: REnv
}

impl Procedure {
    fn new(params: Vec<String>, body: Rc<Token>, outer_env: REnv) -> Procedure {
        Procedure {
            params: params,
            fixed_num: false,
            body: body,
            declarations: vec![],
            outer_env: outer_env
        }
    }

    fn new_with_dec(params: Vec<String>, body: Rc<Token>, declarations: Vec<Rc<Token>>, outer_env: REnv) -> Procedure {
        Procedure {
            params: params,
            fixed_num: false,
            body: body,
            declarations: declarations,
            outer_env: outer_env
        }
    }

    fn new_single(param: String, body: Rc<Token>, outer_env: REnv) -> Procedure {
        Procedure {
            params: vec![param],
            fixed_num: true,
            body: body,
            declarations: vec![],
            outer_env: outer_env
        }
    }

    fn new_single_with_dec(param: String, body: Rc<Token>, declarations: Vec<Rc<Token>>, outer_env: REnv) -> Procedure {
        Procedure {
            params: vec![param],
            fixed_num: true,
            body: body,
            declarations: declarations,
            outer_env: outer_env
        }
    }

    fn call_partial(& self,list: Vec<Rc<Value>>) -> Rc<Value> {
        if !self.fixed_num && self.params.len() != list.len() {
            Rc::new(Value::Error(String::from("parameter length mismatch")))
        } else
        {
            let mut new_env : PartialEnv = PartialEnv::new();
            if self.fixed_num {
                new_env.insert(self.params[0].clone(), Rc::clone(&make_list(list)));
            } else {
                for i in 0..self.params.len() {
                    new_env.insert(self.params[i].clone(),Rc::clone(&list[i]));
                }
            }
            let env = Env::new_sub(new_env, Rc::clone(&self.outer_env));
            if self.declarations.len() > 0 {
                let mut env_rc = Rc::new(env);
                for token in &self.declarations {
                    env_rc = eval_dec(&token, &env_rc);
                }
                Rc::new(Value::Partial(Rc::clone(&self.body), env_rc))
                    //eval_exp(&self.body, &env_rc)
            } else {
                //println!("calling with env {:?}\n",env);
                //eval_exp(&self.body, &Rc::new(env))
                Rc::new(Value::Partial(Rc::clone(&self.body), Rc::new(env)))
            }
        }
    }
}

impl<'a> Clone for Procedure {
    fn clone(&self) -> Procedure {
        Procedure::new(self.params.clone(), self.body.clone(), self.outer_env.clone())
    }
}

/*fn onearg(func: fn(Rc<Value>) -> Rc<Value>) -> Rc<dyn Fn(Vec<Rc<Value>>) -> Rc<Value>>
{
    Rc::new(move |list: Vec<Rc<Value>>| if list.len() == 1 {
        func(Rc::clone(&list[0]))
    } else {
        Rc::new(Value::Error(String::from("wrong number of arguments (expected 1)")))
    })
}*/

fn onearg(list: Vec<Rc<Value>>, func: fn(Rc<Value>) -> Rc<Value>) -> Rc<Value>
{
    if list.len() == 1 {
        func(Rc::clone(&list[0]))
    } else {
        Rc::new(Value::Error(String::from("wrong number of arguments (expected 1)")))
    }
}

fn twoarg(list: Vec<Rc<Value>>, func: fn(Rc<Value>, Rc<Value>) -> Rc<Value>) -> Rc<Value>
{
    if list.len() == 2 {
        func(Rc::clone(&list[0]), Rc::clone(&list[1]))
    } else {
        Rc::new(Value::Error(String::from("wrong number of arguments (expected 1)")))
    }
}

fn to_number_list(list: Vec<Rc<Value>>) -> Option<Vec<i64>> {
    let mut ret_list: Vec<i64> = vec![];
    for outer_value in list.iter() {
        if let Value::Integer(value) = **outer_value {
            ret_list.push(value as i64)
        } else  {
            return None
        }
    }
    Some(ret_list)
}

fn add(list: Vec<Rc<Value>>) -> Rc<Value> {
    twoarg(list, |first, second| {
        match [&*first, &*second] {
            [Value::Integer(f), Value::Integer(s)] => Rc::new(Value::Integer(f + s)),
            _ => Rc::new(Value::Error(String::from("can't add non numbers")))
        }
    })
}

fn mult(list:  Vec<Rc<Value>>) -> Rc<Value> {
    twoarg(list, |first, second| {
        match [&*first, &*second] {
            [Value::Integer(f), Value::Integer(s)] => Rc::new(Value::Integer(f * s)),
            _ => Rc::new(Value::Error(String::from("can't mult non numbers")))
        }
    })
}

fn sub(list: Vec<Rc<Value>>) -> Rc<Value> {
    match to_number_list(list) {
        None => Rc::new(Value::Error(String::from("can't subtract non numbers"))),
        Some(int_list) => match int_list.as_slice() {
            [] => Rc::new(Value::Error(String::from("no number for subtract"))),
            [value] => Rc::new(Value::Integer(-value)),
            [f,s] => Rc::new(Value::Integer(f-s)),
            _ => Rc::new(Value::Error(String::from("unexpected subtract")))
        }
    }
}

fn num_equal(list: Vec<Rc<Value>>) -> Rc<Value> {
    twoarg(list, |first, second| {
        match [&*first, &*second] {
            [Value::Integer(x),Value::Integer(y)] => Rc::new(Value::Boolean(x == y)),
            _ => Rc::new(Value::Error(String::from("can't = on non numbers")))
        }
    })
}

fn num_increase(list: Vec<Rc<Value>>) -> Rc<Value> {
    twoarg(list, |first, second| {
        match [&*first, &*second] {
            [Value::Integer(x),Value::Integer(y)] => Rc::new(Value::Boolean(x < y)),
            _ => Rc::new(Value::Error(String::from("can't < on non numbers")))
        }
    })
}

fn num_decrease(list: Vec<Rc<Value>>) -> Rc<Value> {
    twoarg(list, |first, second| {
        match [&*first, &*second] {
            [Value::Integer(x),Value::Integer(y)] => Rc::new(Value::Boolean(x > y)),
            _ => Rc::new(Value::Error(String::from("can't > on non numbers")))
        }
    })
}

fn make_list<'a>(list:  Vec<Rc<Value>>) -> Rc<Value> {
    let mut cur_pair = Value::EmptyList;
    for value in list.iter().rev() {
        cur_pair = Value::Pair(Rc::clone(value), Rc::new(cur_pair))
    }
    Rc::new(cur_pair)
}

fn make_quote(token: &Token) -> Value {
    match token {
        &Token::IntegerToken(value) => Value::Integer(value),
        &Token::StringToken(ref value) => Value::Symbol(value.clone()),
        &Token::TokenList(ref list) => {
            let mut list_iter = list.iter();
            let mut cur_pair = Value::EmptyList;
            if list.len() >= 3 {
                if let Token::Dot = list[list.len()-2] {
                    if let Some(last) = list_iter.next_back() {
                        cur_pair = make_quote(last);
                        list_iter.next_back(); //Pop Dot off
                    }
                }
            }
            for sub_token in list_iter.rev() {
                cur_pair = Value::Pair(Rc::new(make_quote(sub_token)), Rc::new(cur_pair))
            }
            cur_pair
        }
        &Token::QuoteToken(ref value) => make_quote(value),
        &Token::Dot => Value::Error(String::from("unexpected .")),
    }
}

fn nullp(list: Vec<Rc<Value>>) -> Rc<Value>{
    onearg(list, |value| {
        if let Value::EmptyList = *value {
            Rc::new(Value::Boolean(true))
        } else {
            Rc::new(Value::Boolean(false))
        }
    })
}

fn zerop(list: Vec<Rc<Value>>) -> Rc<Value> {
    onearg(list, |value| {
        if let Value::Integer(x) = *value {
            Rc::new(Value::Boolean(x == 0))
        } else {
            Rc::new(Value::Error(String::from("can't zero? on non single number")))
        }
    })
}

fn not_fn(list: Vec<Rc<Value>>) -> Rc<Value> {
    onearg(list, |value| {
        match *value {
            Value::Boolean(false) => Rc::new(Value::Boolean(true)),
            _ => Rc::new(Value::Boolean(false)),
        }
    })
}

fn car<'a>(list: Vec<Rc<Value>>) -> Rc<Value> {
    onearg(list, |value| {
        match *value {
            Value::Pair(ref car, _) => Rc::clone(car),
            _ => Rc::new(Value::Error(String::from("car has unexpected argument")))
        }
    })
}

fn cdr<'a>(list: Vec<Rc<Value>>) -> Rc<Value> {
    onearg(list, |value| {
        match *value {
            Value::Pair(_, ref cdr) => Rc::clone(cdr),
            _ => Rc::new(Value::Error(String::from("cdr has unexpected argument")))
        }
    })
}

fn cons<'a>(list: Vec<Rc<Value>>) -> Rc<Value> {
    twoarg(list, |car, cdr| {
        Rc::new(Value::Pair(Rc::clone(&car), Rc::clone(&cdr)))
    })
}

fn eqvp(list: Vec<Rc<Value>>) -> Rc<Value> {
    //Note, not completely r7rs compliant.
    match list.as_slice() {
        [first, second] => match [&**first, &**second] {
            [Value::Symbol(ref x),Value::Symbol(ref y)] => Rc::new(Value::Boolean(x == y)),
            [Value::Boolean(x),Value::Boolean(y)] => Rc::new(Value::Boolean(x == y)),
            [Value::EmptyList, Value::EmptyList] => Rc::new(Value::Boolean(true)),
            [Value::Integer(x), Value::Integer(y)] => Rc::new(Value::Boolean(x == y)),
            [_, _] => Rc::new(Value::Boolean(false)),
        }
        _ => Rc::new(Value::Error(String::from("eqv? should have two arguments")))
    }
}

fn numberp(list: Vec<Rc<Value>>) -> Rc<Value> {
    match list.as_slice() {
        [head] => match [&**head] {
            [Value::Integer(_)] => Rc::new(Value::Boolean(true)),
            [_] => Rc::new(Value::Boolean(false)),
        }
        _ => Rc::new(Value::Error(String::from("number? should have one arguments")))
    }
}

fn pairp(list: Vec<Rc<Value>>) -> Rc<Value> {
    onearg(list, |head| {
        match &*head {
            Value::Pair(_, _) => Rc::new(Value::Boolean(true)),
            _ => Rc::new(Value::Boolean(false)),
        }
    })
}

fn booleanp(list: Vec<Rc<Value>>) -> Rc<Value> {
    onearg(list, |head| {
        match &*head {
            Value::Boolean(_) => Rc::new(Value::Boolean(true)),
            _ => Rc::new(Value::Boolean(false)),
        }
    })
}

fn procedurep(list: Vec<Rc<Value>>) -> Rc<Value> {
    onearg(list, |head| {
        match &*head {
            Value::Procedure(_) => Rc::new(Value::Boolean(true)),
            Value::Function(_) => Rc::new(Value::Boolean(true)),
            _ => Rc::new(Value::Boolean(false)),
        }
    })
}

fn pair_to_list(pair: &Value) -> Option<Vec<Rc<Value>>> {
    if let Value::Pair(ref car, ref cdr) = pair {
        let mut cur_cdr = cdr;
        let mut done = false;
        let mut list: Vec<Rc<Value>> = vec![Rc::clone(car)];
        while !done {
            if let Value::Pair(ref cadr, ref cddr) = **cur_cdr {
                list.push(Rc::clone(cadr));
                cur_cdr = cddr
            }
            else if let Value::EmptyList = **cur_cdr {
                done = true
            } else {
                //Not proper list so return None
                return None
            }
        }
        Some(list)
    } else {
        None
    }
}

fn apply(list: Vec<Rc<Value>>) -> Rc<Value> {
    twoarg(list, |first, second| {
        match [&*first, &*second] {
            [Value::Procedure(procedure), pair @ Value::Pair(_, _)] => {
                if let Some(list) = pair_to_list(pair) {
                    procedure.call_partial(list)
                } else {
                    Rc::new(Value::Error(String::from("apply requires function and proper list")))
                }
            },
            [Value::Function(func), pair @ Value::Pair(_, _)] => {
                if let Some(list) = pair_to_list(pair) {
                    func(list)
                } else {
                    Rc::new(Value::Error(String::from("apply requires function and proper list")))
                }
            },
            _ => Rc::new(Value::Error(String::from("apply requires function and list")))
        }
    })
}

fn symbolp(list: Vec<Rc<Value>>) -> Rc<Value> {
    onearg(list, |head| {
        match &*head {
            Value::Symbol(_) => Rc::new(Value::Boolean(true)),
            _ => Rc::new(Value::Boolean(false)),
        }
    })
}

fn if_fn(test: &Token, consequent: &Token, optional_alternate: Option<&Token>, env: &REnv)
         -> Rc<Value>
{
    let test_eval = eval_exp(test,env);
    match *test_eval {
        Value::Error(_) => test_eval,
        Value::Boolean(false) =>
            match optional_alternate {
                Some(alternate) => Rc::new(Value::Partial(Rc::new(alternate.clone()), Rc::clone(env))), //eval_exp(alternate, env),
                None => Rc::new(Value::Undefined)
            },
        _ =>
            Rc::new(Value::Partial(Rc::new(consequent.clone()), Rc::clone(env)))
    }
}

fn eval_both<'a, 'b, 'c>(token: &Token, env: &REnv) -> (REnv, Rc<Value>)
{
      match token {
          &Token::TokenList(ref list) => match list as &[Token] {
              [Token::StringToken(ref string),Token::StringToken(ref _name), ref _raw_var_value] if string == "define" =>
                  (eval_dec(token, env), Rc::new(Value::Undefined)),
              _ => (Rc::clone(env), eval_exp(token, env))
          }
          _ => (Rc::clone(env), eval_exp(token, env))
      }
}

fn use_y_combiner(procedure: &Procedure, name: String) -> Rc<Value> {
    //Needs to generate lambda like:
    //Y1 (lambda (le) ((lambda (f) (f f)) (lambda (f) (le (lambda (x1) ((f f) x1))))))
    //or
    //Y2 (lambda (le) ((lambda (f) (f f)) (lambda (f) (le (lambda (x1 x2) ((f f) x1 x2))))))
    //or
    //(define YS (lambda (le) ((lambda (f) (f f)) (lambda (f) (le (lambda x (apply (f f) x)))))))
    //and then make new procedure
    //(Y1 Original procedure)
    let size = procedure.params.len();
    let args:Vec<Token> = vec![Token::str("#X0")];
    let ff = Token::TokenList(
        vec![Token::str("f"),
             Token::str("f")
             ]);
    let self_call = {
        let mut extra = vec![Token::str("apply"),ff];
        extra.extend(args.clone());
        vec![Token::str("lambda"),
             args[0].clone(),
             Token::TokenList(extra)]
    };
    let new_tokens = Token::TokenList(
        vec![Token::TokenList(
            vec![Token::str("lambda"),
                 Token::TokenList(
                     vec![Token::str("f")]),
                 Token::TokenList(
                     vec![Token::str("f"),
                          Token::str("f")
                          ])
                 ]),
             Token::TokenList(
                 vec![Token::str("lambda"),
                     Token::TokenList(
                         vec![Token::str("f")]),
                     Token::TokenList(
                         vec![Token::str("phi"),
                              Token::TokenList(self_call)
                              ])
                     ])
             ]);
    let mut parameter_tokens = Vec::new();
    for i in 0..size {
        parameter_tokens.push(Token::StringToken(procedure.params[i].clone()));
    }
    // ((lambda (phi) new_tokens) (lambda (name) (lambda (parameters) body)))
    //Now need to create new_tokens as a procedure
    //TODO: Not sure if we should just pass a blank environment here.
    let y_proc = Procedure::new(vec![String::from("phi")], Rc::new(new_tokens), Rc::clone(&procedure.outer_env));
    let new_lambda = if procedure.fixed_num {
        Token::TokenList(vec![Token::str("lambda"),parameter_tokens[0].clone(),(*procedure.body).clone()])
    } else {
        Token::TokenList(vec![Token::str("lambda"),Token::TokenList(parameter_tokens),(*procedure.body).clone()])
    };
    //Evaluate y combiner on original procedure, and return result
    let orig_part = Procedure::new(vec![name.clone()], Rc::new(new_lambda), Rc::clone(&procedure.outer_env)); //Rc::clone(&procedure.body)
    //stuff into environment and eval
    let mut new_env_part : PartialEnv = PartialEnv::new();
    new_env_part.insert(String::from("Y"), Rc::new(Value::Procedure(y_proc)));
    new_env_part.insert(name.clone() + &String::from("part"), Rc::new(Value::Procedure(orig_part)));
    let new_env = Env::new_sub(new_env_part, Rc::clone(&procedure.outer_env));
    let transformed = eval_exp(&Token::TokenList(vec![Token::str("Y"), Token::StringToken(name + &String::from("part"))]), &Rc::new(new_env));
    transformed
}

fn eval_dec<'a, 'b, 'c>(token: &Token, env: &REnv) -> REnv
{
    match token {
        &Token::TokenList(ref list) => match list as &[Token] {
            [Token::StringToken(ref string),Token::StringToken(ref name), ref raw_var_value] if string == "define" => {
                let mut new_env: PartialEnv = PartialEnv::new();
                let var_value = eval_exp(raw_var_value,env);
                if let Value::Procedure(procedure) = &*var_value {
                    new_env.insert(name.clone(), use_y_combiner(procedure, name.to_string()));
                } else {
                    new_env.insert(name.clone(), var_value);
                }
                Rc::new(Env::new_sub(new_env, Rc::clone(env)))
            },
            _ => {
                println!("Internal error, unexpected in define");
                Rc::clone(env)
            }
        }
        _ => {
            println!("Internal error, unexpected in define");
            Rc::clone(env)
        }
    }
}

fn eval_exp<'a,'b, 'c>(token_orig: &Token, env_orig: &REnv)
 -> Rc<Value>
{
    let mut token = Rc::new(token_orig.clone());
    let mut env = Rc::clone(env_orig);
    loop {
        //println!("eval_exp {:?}", token);
        let eval = match &*token {
            &Token::Dot => Rc::new(Value::Error(String::from("unexpected ."))),
            &Token::IntegerToken(value) => Rc::new(Value::Integer(value)),
            &Token::StringToken(ref value) => match env.find(&value) {
                Some(entry) => entry,
                None => {//println!("Can't find {:?} in {:?}",value, env);
                    Rc::new(Value::Error(String::from("token not found in env")))}
            },
            &Token::QuoteToken(ref value) => Rc::new(make_quote(value)),
            &Token::TokenList(ref list) => match list as &[Token] {
                [Token::StringToken(ref string), ref test, ref consequent, ref alternate] if string == "if" => {
                    if_fn(test, consequent, Some(alternate), &env)
                },
                [Token::StringToken(ref string), ref test, ref consequent] if string == "if" => {
                    if_fn(test, consequent, None, &env)
                },
                [Token::StringToken(ref string),..] if string == "if"  => Rc::new(Value::Error(String::from("if missing a bit"))),
                [Token::StringToken(ref string), Token::TokenList(ref test1), ..] if string == "cond" && test1.len() == 2 =>
                {
                    let mut in_else = false;
                    if let Token::StringToken(ref cond_start) = test1[0]  {
                        in_else = cond_start == "else";
                    }
                    if in_else {
                        eval_exp(&test1[1], &env)
                    } else if list.len() > 2 {
                        let mut rest:Vec<Token>  = Vec::new();
                        rest.push(Token::str("cond"));
                        rest.extend_from_slice(&list[2..]);
                        if_fn(&test1[0], &test1[1], Some(&Token::TokenList(rest)), &env)
                    } else {
                        if_fn(&test1[0], &test1[1], None, &env)
                    }
                },
                [Token::StringToken(ref string)] if string == "and" =>
                    Rc::new(Value::Boolean(true)),
                [Token::StringToken(ref string), ref test1] if string == "and" => eval_exp(test1, &env),
                [Token::StringToken(ref string), ref test1, ..] if string == "and" =>
                {
                    let mut rest:Vec<Token> = Vec::new();
                    rest.push(Token::str("and"));
                    rest.extend_from_slice(&list[2..]);
                    if_fn(&test1, &Token::TokenList(rest), Some(&Token::str("#f")), &env)
                },
                [Token::StringToken(ref string)] if string == "or" =>
                    Rc::new(Value::Boolean(false)),
                [Token::StringToken(ref string), ref test1] if string == "or" =>
                    eval_exp(test1, &env),
                [Token::StringToken(ref string), ref test1, rest @ ..] if string == "or" =>
                {
                    let mut transformed:Vec<Token> = vec![Token::str("let")];
                    transformed.push(Token::TokenList(vec![
                        Token::TokenList(vec![Token::str("#x"), test1.clone()])]));
                    let mut rest_or:Vec<Token> = vec![Token::str("or")];
                    for rest_item in rest {
                        rest_or.push(rest_item.clone());
                    }
                    transformed.push(Token::TokenList(vec![
                        Token::str("if"),Token::str("#x"),Token::str("#x"),
                        Token::TokenList(rest_or)]));
                    eval_exp(&Token::TokenList(transformed), &env)
                }
                [Token::StringToken(ref string), Token::StringToken(ref param), ref body] if string == "lambda" => {
                    Rc::new(Value::Procedure(Procedure::new_single(String::from(param),Rc::new(body.clone()),env.clone())))
                }
                [Token::StringToken(ref string), Token::StringToken(ref param), middle @ .., ref body] if string == "lambda" => {
                    let mut middle_vec: Vec<Rc<Token>> = Vec::new();
                    for dec in middle {
                        middle_vec.push(Rc::new(dec.clone()));
                    }
                    Rc::new(Value::Procedure(Procedure::new_single_with_dec(String::from(param),Rc::new(body.clone()),middle_vec,env.clone())))
                }
                [Token::StringToken(ref string),Token::TokenList(ref parameters), ref body]
                    if string == "lambda" => {
                        let mut param_eval: Vec<String> = vec![];
                        for item in parameters.iter() {
                            match item as &Token {
                                &Token::StringToken(ref param) => param_eval.push(param.clone()),
                                _ => return Rc::new(Value::Error(String::from("Non string in lambda parameter list")))
                            }
                        }
                        Rc::new(Value::Procedure(Procedure::new(param_eval,Rc::new(body.clone()),env.clone())))
                    },
                [Token::StringToken(ref string),Token::TokenList(ref parameters), middle @ .., ref body]
                    if string == "lambda" => {
                        let mut middle_vec: Vec<Rc<Token>> = Vec::new();
                        for dec in middle {
                            middle_vec.push(Rc::new(dec.clone()));
                        }
                        let mut param_eval: Vec<String> = vec![];
                        for item in parameters.iter() {
                            match item as &Token {
                                &Token::StringToken(ref param) => param_eval.push(param.clone()),
                                _ => return Rc::new(Value::Error(String::from("Non string in lambda parameter list")))
                            }
                        }
                        Rc::new(Value::Procedure(Procedure::new_with_dec(param_eval,Rc::new(body.clone()), middle_vec, env.clone())))
                    },
                [Token::StringToken(ref string),datum] if string == "quote" => {
                    Rc::new(make_quote(datum))
                }
                [Token::StringToken(ref string), Token::TokenList(var_inits), body @ ..] if string == "let" => {
                    let mut vars: Vec<Token> = Vec::new();
                    let mut inits: Vec<Token> = Vec::new();
                    for var_init in var_inits {
                        if let Token::TokenList(inner_list) = var_init {
                            vars.push(inner_list[0].clone());
                            inits.push(inner_list[1].clone());
                        } else {
                            return  Rc::new(Value::Error(String::from("invalid let var init")))
                        }
                    }
                    let mut transformed: Vec<Token> = Vec::new();
                    let mut lambda_transform: Vec<Token> = Vec::new();
                    lambda_transform.push(Token::str("lambda"));
                    lambda_transform.push(Token::TokenList(vars));
                    for body_item in body {
                        lambda_transform.push(body_item.clone());
                    }
                    transformed.push(Token::TokenList(lambda_transform));
                    transformed.extend(inits);
                    eval_exp(&Token::TokenList(transformed), &env)
                }
                [] => Rc::new(Value::Undefined),
                full_token =>
                    if let Some((head, tail)) = full_token.split_first() {
                        //[ref head, tail..] => {
                        //println!("Evaluating {:?} in &env {:?}\n",head,&env);
                        let head_eval = eval_exp(head,&env);
                        //println!("Evaluated {:?} in env {:?}\n",head_eval,&env);
                        //tail.iter().map(|t| eval_exp(t,&env).expect("Got none")).collect();
                        let mut tail_eval: Vec<Rc<Value>> = vec![];
                        for item in tail.iter() {
                            let ret = eval_exp(item,&env);
                            //println!("debug {:?}",item);
                            tail_eval.push(ret);
                        }
                        //let tail1_eval = eval_exp(tail1, &env).expect("Got none1");
                        match &*head_eval {
                            Value::Function(func) =>
                                func(tail_eval),
                            Value::Procedure(procedure) =>
                                procedure.call_partial(tail_eval),
                            _ => head_eval
                        }
                    } else {
                        Rc::new(Value::Error(String::from("impossible")))
                    }
            }
        };
        if let Value::Partial(ref sub_token, ref sub_env) = *eval {
            env = Rc::clone(&sub_env);
            token = sub_token.clone();
        } else {
            return eval;
        }
    }
}


fn main() {
    let mut lines = vec![];
    //let env: RREnv = Rc::new(RefCell::new(Env::new()));
    let mut init_env: PartialEnv = PartialEnv::new();
    init_env.insert(String::from("+"),Rc::new(Value::Function(add)));
    init_env.insert(String::from("*"),Rc::new(Value::Function(mult)));
    init_env.insert(String::from("-"),Rc::new(Value::Function(sub)));
    init_env.insert(String::from("="),Rc::new(Value::Function(num_equal)));
    init_env.insert(String::from("<"),Rc::new(Value::Function(num_increase)));
    init_env.insert(String::from(">"),Rc::new(Value::Function(num_decrease)));
    init_env.insert(String::from("list"),Rc::new(Value::Function(make_list)));
    init_env.insert(String::from("car"),Rc::new(Value::Function(car)));
    init_env.insert(String::from("cdr"),Rc::new(Value::Function(cdr)));
    init_env.insert(String::from("cons"),Rc::new(Value::Function(cons)));
    init_env.insert(String::from("null?"), Rc::new(Value::Function(nullp)));
    init_env.insert(String::from("zero?"),Rc::new(Value::Function(zerop)));
    init_env.insert(String::from("not"),Rc::new(Value::Function(not_fn)));
    init_env.insert(String::from("eqv?"), Rc::new(Value::Function(eqvp)));
    init_env.insert(String::from("number?"), Rc::new(Value::Function(numberp)));
    init_env.insert(String::from("pair?"), Rc::new(Value::Function(pairp)));
    init_env.insert(String::from("boolean?"), Rc::new(Value::Function(booleanp)));
    init_env.insert(String::from("procedure?"), Rc::new(Value::Function(procedurep)));
    init_env.insert(String::from("symbol?"), Rc::new(Value::Function(symbolp)));
    init_env.insert(String::from("apply"), Rc::new(Value::Function(apply)));
    init_env.insert(String::from("#t"),Rc::new(Value::Boolean(true)));
    init_env.insert(String::from("#f"),Rc::new(Value::Boolean(false)));
    //println!("initial env {:?}",env);

    let mut env: REnv = Rc::new(Env::new(init_env));
    let mut current_parse = String::new();

    loop {
        if current_parse.len() == 0 {
            print!("> ");
            let _ = std::io::stdout().flush();
        }
        let mut line = String::new();

        let input_result = io::stdin()
            .read_line(&mut line);

        match input_result {
            Ok(0) => break,
            Err(e) => {println!("Error: {}",e); break},
            _ => {
                lines.push(line);
                let last_line = lines.last().expect("line");
                current_parse.push_str(last_line);
                match parse(&current_parse) {
                    ParseOption::Some(ref parsed) => {
                        let (new_env, value) = eval_both(parsed,&env);
                        env = new_env;
                        println!("{:?}",value);
                        current_parse.clear();
                    },
                    ParseOption::Partial => { /*should get more input*/ },
                    ParseOption::None => {
                        println!("Nothing");
                        current_parse.clear();
                    }
                }
            }
        }
    }
    println!("");
}
