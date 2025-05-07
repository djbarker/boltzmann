# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile build

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Run the apidoc step to generate the API documentation before calling Sphynx.
# Note: the paths following ../python/boltzmann are excludes.
build:
	$(MAKE) clean
	sphinx-apidoc -ef \
		-o source/api \
		--templatedir=source/_templates \
		../python/boltzmann \
		../python/boltzmann/kernel_gen.py \
		../python/boltzmann/boltzmann.*so
	$(MAKE) html

# See README.md
deploy:
	$(MAKE) build
	cd ..
	cp -r docs/build/html/* .
	git add -u
	git commit -m "Update docs."
	git push