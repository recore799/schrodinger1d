;; -*- lexical-binding: t; -*-

(TeX-add-style-hook
 "numerov"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "11pt")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8") ("fontenc" "T1") ("graphicx" "") ("longtable" "") ("wrapfig" "") ("rotating" "") ("ulem" "normalem") ("amsmath" "") ("amssymb" "") ("capt-of" "") ("hyperref" "") ("amsfonts" "") ("physics" "") ("bm" "") ("geometry" "")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art11"
    "inputenc"
    "fontenc"
    "graphicx"
    "longtable"
    "wrapfig"
    "rotating"
    "ulem"
    "amsmath"
    "amssymb"
    "capt-of"
    "hyperref"
    "amsfonts"
    "physics"
    "bm"
    "geometry")
   (LaTeX-add-labels
    "sec:org76ee3a1"
    "eq:numerov-eq"
    "eq:numerov"
    "sec:org30d4a9f"
    "eq:schr1"
    "eq:ho"
    "sec:orgb1207c5"
    "sec:org2aec1f7"
    "eq:ho_sol1"
    "eq:rec-hermite"
    "eq:ho-energy"
    "sec:org54a3480"
    "sec:org43e3e09"
    "sec:org229c095"
    "sec:orgc35045b"
    "sec:org284b16d"
    "sec:org37ddb88"
    "sec:org0b6b99d"
    "sec:orgfabd782"
    "sec:org24d8140"
    "sec:org59863da"
    "sec:org5621cae"
    "sec:org9ef6a09"
    "sec:orgd4b8e6e"
    "sec:orge2b6a34"
    "sec:orgff6d996"
    "sec:orgb8249d7"
    "sec:orgd7757e0"
    "sec:org6074ade"
    "sec:org1fad292"
    "sec:org76898f5"
    "sec:orgccebd4c"
    "sec:org43f0543"
    "sec:org12980c6"
    "sec:org388d9bc"
    "sec:org91df493"
    "sec:orgc22ce97"
    "eq:schr-rad"
    "sec:orgf89e958"
    "sec:orgb7931cc"))
 :latex)

