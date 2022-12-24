# rust_pr7rs
A partial implementation of r7rs-pico (previously named r7rs-tiny)

This was created as part of r7rs-pico for testing various parts of the
semantics.

Major differences between r7rs-pico is this is not properly tail
recursive (tail recursion uses heap space), and there are various
differences in the syntax.

Integers are 64 bit signed integers, and overflow of +,-,* will return #f

Integer constants that are too big will be renamed to invalid_number_
followed by the number.

The r7rs-pico specification is at: https://github.com/jrincayc/r7rs-pico-spec
