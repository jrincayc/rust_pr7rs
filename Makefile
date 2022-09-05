
rust_pr7rs: rust_pr7rs.rs
	rustc -g --edition 2021 rust_pr7rs.rs

test_pr7rs_out: rust_pr7rs test_pr7rs
	./rust_pr7rs < test_pr7rs > test_pr7rs_out
	./rust_pr7rs hello.scm >> test_pr7rs_out
	diff -u test_pr7rs_expected test_pr7rs_out

clean:
	rm rust_pr7rs test_pr7rs_out
