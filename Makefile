# This is the makefile for installing TADM. See the file
# docs/install.txt for directions on installing TADM.
#
ALL: all

all:
	$(MAKE) -C src

clean:
	rm -f src/*.o
