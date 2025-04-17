# C/C++ Makefile Preset
# -
# Created by Linux-Tech-Tips (Martin), more at https://github.com/Linux-Tech-Tips
# -
# Feel free to use this for any projects, please credit me if you do
# -
# Documentation about the configuration of this file can be found at (https://github.com/Linux-Tech-Tips/Makefile_preset.git)
# -
# NO WARRANTY

# === User-defined variables ===

# USER-DEFINED SECTION OF THE CODE

# General project settings
PROJECT_NAME := Function Approximator
OUTFILE := func.elf

# Compiler and linker programs
CC := gcc
LD := gcc

# C/C++ Language settings
SOURCE_EXT := c
OBJECT_EXT := o
DEPENDENCY_EXT := dep

# C/C++ Compiler and linker settings
COMPILE_FLAG := -c
OUTPUT_FLAG := -o
DEPENDENCY_FLAG := -MM
DEP_TARGET_FLAG := -MT

# Directories
BUILD_DIR := build
SOURCE_DIR := src
DEPS_DIR := .dependencies
DOCS_DIR := doxygen_doc

# Additional compiler/linker flags
CFLAGS := -std=gnu99 -Wall -Werror -pedantic
LDFLAGS := -lc -lm

# Documentation
DOCS_SW := doxygen
DOCS_CONF := Doxyfile

# Makefile shell configuration
SHELL := sh
MKDIR_COMMAND := mkdir -p
RM_COMMAND := rm -r
ECHO_COMMAND := echo

# Makefile meta configuration
DIR_TARGET := dirs
COMPILE_TARGET := compile
LINK_TARGET := link
RUN_TARGET := run
DOCS_TARGET := doc
CLEAN_TARGET := clean
HELP_TARGET := help
ALL_TARGET := $(DIR_TARGET) $(COMPILE_TARGET) $(LINK_TARGET)

# Additional help information
USER_HELP := 


# END OF USER-DEFINED SECTION OF THE CODE
# 	for simple applications, the further code shouldn't need to be edited
# 	feel free to change anything even after this, though, if it's desirable for your use-case


# === Additional pre-defined variables ===

# Address sanitizer compiler/linker flags (if enabled)
ifeq ($(address), true)
	CFLAGS += -fsanitize=address -g
	LDFLAGS += -fsanitize=address -g
endif

# These variables are needed for the compile and link PHONY targets - to know which implicit rules to call
SOURCES := $(shell find $(SOURCE_DIR) -name *.$(SOURCE_EXT))
OBJECTS := $(patsubst %.$(SOURCE_EXT),$(BUILD_DIR)/%.$(OBJECT_EXT),$(notdir $(SOURCES)))

# Search path for the compilation pattern rule
VPATH := $(shell find $(SOURCE_DIR) -type d)


# === Targets ===

all: $(ALL_TARGET)

$(OUTFILE): $(OBJECTS)
	$(CC) $(CFLAGS) $^ $(OUTPUT_FLAG) $@ $(LDFLAGS)

# Pattern rule for compilation (+ generating dependency rules if not forbidden)
$(BUILD_DIR)/%.$(OBJECT_EXT): %.$(SOURCE_EXT)
	$(CC) $(CFLAGS) $(COMPILE_FLAG) $< $(OUTPUT_FLAG) $@
ifneq ($(NO_DEPS), true)
	$(CC) $(CFLAGS) $(DEP_TARGET_FLAG) $@ $(DEPENDENCY_FLAG) $< > $(DEPS_DIR)/$(notdir $@).$(DEPENDENCY_EXT)
endif

.PHONY: $(DIR_TARGET), $(COMPILE_TARGET), $(LINK_TARGET), $(RUN_TARGET), $(DOCS_TARGET), $(CLEAN_TARGET), $(HELP_TARGET)

# SOURCE_DIR is not made, but expected to already exist
# If not specifically forbidden, creates the dependency targets directory as well
ifneq ($(NO_DEPS),true)
$(DIR_TARGET): $(BUILD_DIR) $(DOCS_DIR) $(DEPS_DIR) | $(SRC_DIR)
else
$(DIR_TARGET): $(BUILD_DIR) $(DOCS_DIR) | $(SRC_DIR)
endif

# Specific directory targets
$(BUILD_DIR):
	$(MKDIR_COMMAND) $(BUILD_DIR)
$(DOCS_DIR):
	$(MKDIR_COMMAND) $(DOCS_DIR)
$(DEPS_DIR):
	$(MKDIR_COMMAND) $(DEPS_DIR)

$(COMPILE_TARGET): $(OBJECTS)

$(LINK_TARGET): $(OUTFILE)

$(RUN_TARGET): all
	./$(OUTFILE)

$(DOCS_TARGET):
	$(DOCS_SW) $(DOCS_CONF)

# If not specifically forbidden, cleans everything including dependency rules
$(CLEAN_TARGET):
	-$(RM_COMMAND) $(BUILD_DIR)
	-$(RM_COMMAND) $(OUTFILE)
	-$(RM_COMMAND) $(DOCS_DIR)
ifneq ($(NO_DEPS), true)
	-$(RM_COMMAND) $(DEPS_DIR)
endif

$(HELP_TARGET):
	@$(ECHO_COMMAND) "$(PROJECT_NAME) Makefile: usage: make [target] [options]"
	@$(ECHO_COMMAND) "  - Available targets: $(DIR_TARGET), $(COMPILE_TARGET), $(LINK_TARGET), $(RUN_TARGET), $(DOCS_TARGET), $(CLEAN_TARGET), $(HELP_TARGET), all (default, calls: $(ALL_TARGET))"
	@$(ECHO_COMMAND) "  - Available options:"
	@$(ECHO_COMMAND) "    - address=true - turn on address sanitizer"
	@$(ECHO_COMMAND) "    - NO_DEPS=true - turn off gcc dependency info generation"
	@$(ECHO_COMMAND) "  - Documentation information:"
	@$(ECHO_COMMAND) "    - The target $(DOCS_TARGET) generates code documentation using $(DOCS_SW), which is output into the directory $(DOCS_DIR)"
	@$(ECHO_COMMAND) ""
ifneq ($(USER_HELP),)
	@$(ECHO_COMMAND) $(USER_HELP)
	@$(ECHO_COMMAND) ""
endif
	@$(ECHO_COMMAND) "Makefile preset created by Linux-Tech-Tips (Martin), more at https://github.com/Linux-Tech-Tips"


-include $(DEPS_DIR)/*.$(DEPENDENCY_EXT)
