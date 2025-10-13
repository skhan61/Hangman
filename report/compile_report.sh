#!/bin/bash
# Compile LaTeX report to PDF

set -e

echo "Compiling Hangman Project Report..."
echo "=================================="

# First pass
echo "Running pdflatex (pass 1/2)..."
pdflatex -interaction=nonstopmode hangman_project_report.tex > /dev/null 2>&1

# Second pass for TOC and references
echo "Running pdflatex (pass 2/2)..."
pdflatex -interaction=nonstopmode hangman_project_report.tex > /dev/null 2>&1

echo "=================================="
echo "✓ PDF successfully generated!"
echo ""
ls -lh hangman_project_report.pdf
echo ""
echo "Output: $(pwd)/hangman_project_report.pdf"
echo "Pages: $(pdfinfo hangman_project_report.pdf 2>/dev/null | grep Pages | awk '{print $2}')"
echo ""
echo "Cleaning up auxiliary files..."
rm -f *.aux *.log *.out *.toc *.nav *.snm

echo "✓ Done!"
