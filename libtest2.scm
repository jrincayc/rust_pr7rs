(define-library (extra list)
  (export ellist? ellength)
  (import (scheme base))
  (begin
    (define ellist? (lambda (l)
		      (cond ((null? l) #t)
			    ((not (pair? l)) #f)
			    (else (ellist? (cdr l)))
			    )
		      ))
    (define length_tail (lambda (l so_far)
			       (if (null? l) so_far (length_tail (cdr l) (+ 1 so_far)))

			       ))
    (define ellength (lambda (l) (length_tail l 0)))
    )
  )
(import (extra list))
(display '(list example))
(newline)
(display (ellist? '(a b)))
(newline)
(display (ellist? 'a))
(newline)
(display (ellength '(a b)))
(newline)
