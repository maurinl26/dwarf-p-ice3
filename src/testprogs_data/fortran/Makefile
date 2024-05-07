# Disable the default rules
MAKEFLAGS += --no-builtin-rules --no-builtin-variables

# Project name
NAME := ice_adjust

# Configuration settings
FC := gfortran
AR := ar rcs
LD := $(FC)
RM := rm -f

# List of all source files
SRCS := aux/modd_io.F90 \
		aux/modd_parameters.F90 \
		fiat/parkind1.F90 \
		io/getdata_ice_adjust_mod.F90 \
		io/writedata_ice_adjust_mod.F90 \
		support/arrays_manip.F90 \
		support/stack_mod.F90 \
		support/xrd_getoptions.F90 \
		support/xrd_unix_env.F90
		ice_adjust/main_ice_adjust.F90
# TEST_SRCS := src/tests/csv_read_test.f90 \
#              src/tests/csv_test.f90 \
#              src/tests/csv_write_test.f90

# Create lists of the build artefacts in this project
OBJS := $(addsuffix .o, $(SRCS))
TEST_OBJS := $(addsuffix .o, $(TEST_SRCS))
LIB := $(patsubst %, lib%.a, $(NAME))
TEST_EXE := $(patsubst %.f90, %.exe, $(TEST_SRCS))

# Declare all public targets
.PHONY: all clean
all: $(LIB) $(TEST_EXE)

# Create the static library from the object files
$(LIB): $(OBJS)
	$(AR) $@ $^

# Link the test executables
$(TEST_EXE): %.exe: %.f90.o $(LIB)
	$(LD) -o $@ $^

# Create object files from Fortran source
$(OBJS) $(TEST_OBJS): %.o: %
	$(FC) -c -o $@ $<

# Define all module interdependencies
modd_io.mod := aux/modd_io.F90.o
modd_parameters.mod := aux/modd_parameters.F90.o
parkind1.mod := fiat/parkind1.F90.o
arrays_manip.mod := support/arrays_manip.F90.o
stack_mod.mod := support/stack_mod.F90.o
xrd_getoptions := support/xrd_getoptions.F90.o
xrd_unix_env := support/xrd_unix_env/F90.o
getdata_ice_adjust_mod.mod := io/getdata_ice_adjust_mod.F90.o
writedata_ice_adjust_mod.mod := io/writedata_ice_adjust_mod.F90.o


ice_adjust/main_ice_adjust.F90.o: $(xrd_getoptions.mod)
ice_adjust/main_ice_adjust.F90.o: $(getdata_ice_adjust_mod.mod)
ice_adjust/main_ice_adjust.F90.o: $(writedata_ice_adjust_mod.mod)
ice_adjust/main_ice_adjust.F90.o: $(modd_io.mod)
ice_adjust/main_ice_adjust.F90.o: $(stack_mod.mod)

aux/modd_io.F90.o: $(stack_mod.mod)

io/getdata_ice_adjust_mod.F90.o: $(arrays_manip.mod)
io/getdata_ice_adjust_mod.F90.o: $(parkind1.mod)

io/writedata_ice_adjust_mod.F90.o: $(arrays_manip.mod)
io/writedata_ice_adjust_mod.F90.o: $(parkind1.mod)

support/xrd_getoptions.F90.o: $(parkind1.mod)
support/xrd_getoptions.F90.o: $(xrd_unix_env.mod)

support/xrd_unix_env.F90.o: $(parkind1.mod)



# Cleanup, filter to avoid removing source code by accident
clean:
	$(RM) $(filter %.o, $(OBJS) $(TEST_OBJS)) $(filter %.exe, $(TEST_EXE)) $(LIB) $(wildcard *.mod)
