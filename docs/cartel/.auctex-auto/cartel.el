;; -*- lexical-binding: t; -*-

(TeX-add-style-hook
 "cartel"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("beamer" "final")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("beamerposter" "size=a0" "scale=1.2") ("graphicx" "") ("amsmath" "") ("amssymb" "") ("qrcode" "") ("multicol" "") ("booktabs" "")))
   (TeX-run-style-hooks
    "latex2e"
    "beamer"
    "beamer10"
    "beamerposter"
    "graphicx"
    "amsmath"
    "amssymb"
    "qrcode"
    "multicol"
    "booktabs"))
 :latex)

