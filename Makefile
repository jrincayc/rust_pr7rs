
rust_tr7rs: rust_tr7rs.rs
	rustc -g --edition 2021 rust_tr7rs.rs

test_tr7rs_out: rust_tr7rs test_tr7rs
	./rust_tr7rs < test_tr7rs > test_tr7rs_out
	diff -u test_tr7rs_expected test_tr7rs_out
