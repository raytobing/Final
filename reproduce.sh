# Run the noetbook and create figures
ipython ConsLaborModel.py

# compile latex file
pdflatex LiqConstr.tex
bibtex LiqConstr
pdflatex LiqConstr.tex
pdflatex LiqConstr.tex
rm LiqConstr.aux LiqConstr.bbl LiqConstr.blg LiqConstr.dep LiqConstr.log LiqConstr.out
