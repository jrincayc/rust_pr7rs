(define lib-assoc-list
  (cons
   (cons '(extra list)
	 (let ()
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
	   (filter-env (#env) '(ellist? ellength))
	   )
	 )
   lib-assoc-list
   )
  )
(display 'Created-environment)
(newline)
(display lib-assoc-list)
(newline)
(display (asslc '(extra list) lib-assoc-list))
(newline)
;(merge-env-in (asslc '(extra list) lib-assoc-list))
(import (extra list))
(display '(list example))
(newline)
(display (ellist? '(a b)))
(newline)
(display (ellist? 'a))
(newline)
(display (ellength '(a b)))
(newline)