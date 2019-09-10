PYSRC=$(wildcard *.py)

index.html: README.md header.html
	cp assets/header.html tmp
	pandoc -t html5 -f markdown_github-hard_line_breaks+yaml_metadata_block+markdown_in_html_blocks+auto_identifiers --highlight-style=breezedark README.md >> tmp
	mv tmp index.html
# You need to run `pandoc -s` to get the CSS if you change the highlight style... stupid pandoc. https://stackoverflow.com/q/57128572/500207

%.py: README.md
	node md2code.js