index.html: README.md assets/header.html
	cp assets/header.html tmp
	pandoc -t html5 -f markdown_github-hard_line_breaks+yaml_metadata_block+markdown_in_html_blocks+gfm_auto_identifiers --highlight-style=breezedark README.md >> tmp
	mv tmp index.html
