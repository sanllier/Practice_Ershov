all: build

#---------------------------------------------------------------------------------

OBJDIR = obj/
BINDIR = bin/
INCDIR = ../common/

#---------------------------------------------------------------------------------

CC = mpicxx
DFLAG =
CFLAG = -std=c++11 -I$(INCDIR) -I$(INCDIR)/parparser/

#---------------------------------------------------------------------------------

BINFILE = $(notdir $(shell pwd))

FILES = ../common/parparser/parparser main

CFILES = $(addsuffix .cpp, $(FILES)))
OBJECTS = $(addprefix $(OBJDIR), $(notdir $(addsuffix .o, $(FILES))))

#---------------------------------------------------------------------------------

build: $(OBJECTS)
	@mkdir -p $(BINDIR)
	@echo "\033[30;1;41m "$(BINDIR)" dir was created \033[0m"
	@$(CC) $(OBJECTS) -o $(addsuffix $(BINFILE), $(BINDIR))
	@echo "\033[30;1;41m Builded successfully! \033[0m"
	@echo "\033[30;1;41m --> $(BINDIR)$(BINFILE) \033[0m"

$(OBJDIR)%.o: %.cpp
	@mkdir -p $(OBJDIR)
	@$(CC) -c $(DFLAG) $(CFLAG) $^ -o $@
	@echo "\033[30;1;46m $^ -> $@ \033[0m\n"

obj/parparser.o: ../common/parparser/parparser.cpp
	@mkdir -p $(OBJDIR)
	@$(CC) -c $(DFLAG) $(CFLAG) $^ -o $@
	@echo "\033[30;1;46m $^ -> $@ \033[0m\n"

clean:
	rm -r -f $(OBJDIR)
	rm -r -f $(BINDIR)

#---------------------------------------------------------------------------------
