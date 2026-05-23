//! `mldsa-cli` — CLI for the ML-DSA-65 bridge.
//!
//! Single binary that exposes three sub-commands:
//!
//! ```text
//! mldsa-cli keygen --pk-out PATH --sk-out PATH
//! mldsa-cli sign   --sk PATH --msg PATH --sig-out PATH
//! mldsa-cli verify --pk PATH --msg PATH --sig PATH      # exit 0 on valid, 1 on invalid
//! ```
//!
//! All I/O is via files (binary), to make it easy to call from
//! Python's `subprocess.run` and to keep the cross-language test
//! agnostic to encoding details.
//!
//! No `clap` dependency: argument parsing is hand-rolled to keep the
//! binary tiny and the Cargo dependency tree minimal.

use std::env;
use std::fs;
use std::path::Path;
use std::process::ExitCode;

use mldsa_bridge::{keygen, sign, verify};

fn usage() -> ExitCode {
    eprintln!(
        "\
mldsa-cli — ML-DSA-65 (FIPS 204) bridge for arya-STARK

USAGE:
    mldsa-cli keygen --pk-out PATH --sk-out PATH
    mldsa-cli sign   --sk PATH --msg PATH --sig-out PATH
    mldsa-cli verify --pk PATH --msg PATH --sig PATH

verify exits with status 0 on valid, 1 on invalid, 2 on usage error."
    );
    ExitCode::from(2)
}

fn parse_flag<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    let mut iter = args.iter();
    while let Some(a) = iter.next() {
        if a == flag {
            return iter.next().map(|s| s.as_str());
        }
    }
    None
}

fn cmd_keygen(args: &[String]) -> ExitCode {
    let pk_out = match parse_flag(args, "--pk-out") {
        Some(p) => p,
        None => return usage(),
    };
    let sk_out = match parse_flag(args, "--sk-out") {
        Some(p) => p,
        None => return usage(),
    };
    match keygen() {
        Ok((pk, sk)) => {
            if let Err(e) = fs::write(Path::new(pk_out), &pk) {
                eprintln!("keygen: cannot write pk to {pk_out}: {e}");
                return ExitCode::from(2);
            }
            if let Err(e) = fs::write(Path::new(sk_out), &sk) {
                eprintln!("keygen: cannot write sk to {sk_out}: {e}");
                return ExitCode::from(2);
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("keygen failed: {e}");
            ExitCode::from(2)
        }
    }
}

fn cmd_sign(args: &[String]) -> ExitCode {
    let sk_path = match parse_flag(args, "--sk") {
        Some(p) => p,
        None => return usage(),
    };
    let msg_path = match parse_flag(args, "--msg") {
        Some(p) => p,
        None => return usage(),
    };
    let sig_out = match parse_flag(args, "--sig-out") {
        Some(p) => p,
        None => return usage(),
    };
    let sk = match fs::read(sk_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("sign: cannot read sk from {sk_path}: {e}");
            return ExitCode::from(2);
        }
    };
    let msg = match fs::read(msg_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("sign: cannot read msg from {msg_path}: {e}");
            return ExitCode::from(2);
        }
    };
    match sign(&sk, &msg) {
        Ok(sig) => {
            if let Err(e) = fs::write(Path::new(sig_out), &sig) {
                eprintln!("sign: cannot write sig to {sig_out}: {e}");
                return ExitCode::from(2);
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("sign failed: {e}");
            ExitCode::from(2)
        }
    }
}

fn cmd_verify(args: &[String]) -> ExitCode {
    let pk_path = match parse_flag(args, "--pk") {
        Some(p) => p,
        None => return usage(),
    };
    let msg_path = match parse_flag(args, "--msg") {
        Some(p) => p,
        None => return usage(),
    };
    let sig_path = match parse_flag(args, "--sig") {
        Some(p) => p,
        None => return usage(),
    };
    let pk = match fs::read(pk_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("verify: cannot read pk: {e}");
            return ExitCode::from(2);
        }
    };
    let msg = match fs::read(msg_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("verify: cannot read msg: {e}");
            return ExitCode::from(2);
        }
    };
    let sig = match fs::read(sig_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("verify: cannot read sig: {e}");
            return ExitCode::from(2);
        }
    };
    match verify(&pk, &msg, &sig) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("verify error: {e}");
            ExitCode::from(2)
        }
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        return usage();
    }
    let cmd = args[1].as_str();
    let rest = &args[2..].to_vec();
    match cmd {
        "keygen" => cmd_keygen(rest),
        "sign" => cmd_sign(rest),
        "verify" => cmd_verify(rest),
        _ => usage(),
    }
}
