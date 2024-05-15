# COMPIL = gfortran
COMPIL = mpif90
FLAGS  = -ffree-line-length-none	\
		-fno-protect-parens	\
		-g3 -fcheck=bounds	\
		-funroll-loops	\
		-march=native	\
		-fbackslash	\
		-ffast-math -O3	\
		-flto	\
        -fno-strict-aliasing \
		-mcmodel=large
EXEDIR = ..
OBJS   = modd_io.o     \
        modd_parameters.o	\
	 	parkind1.o	\
	 	writedata_ice_adjust_mod.o	\
	 	xrd_getoptions.o      \
	 	xrd_unix_env.o	\
		arrays_manip.o

EXE    = main_ice_adjust
#---------------------------------------------------------------------#

#-------------  Compilation et creation de l'executable  -------------#

all: $(OBJS) main_ice_adjust.o
	@$(COMPIL) $(FLAGS) -o $(EXEDIR)/$(EXE) $(OBJS) main_ice_adjust.o -L. -L/usr/lib -I/usr/include -lnetcdff
	@echo -n " "
	@echo -n "\n Succès de la compilation à "
	@date +%R
	@echo " "

#---------------------------------------------------#
#------------  Compilation des modules  ------------#

main_ice_adjust.o: main_ice_adjust.F90 $(OBJS)
	@echo -n "main_ice_adjust.o      : "
	@$(COMPIL) $(FLAGS) -c main_ice_adjust.F90
	@echo "    ok"

writedata_ice_adjust_mod.o: writedata_ice_adjust_mod.F90 parkind1.o arrays_manip.o
	@echo -n "writedata_ice_adjust_mod.o      : "
	@$(COMPIL) $(FLAGS) -c writedata_ice_adjust_mod.F90 -lnetcdff -I/usr/include
	@echo "    ok"

arrays_manip.o: arrays_manip.F90 parkind1.o
	@echo -n "arrays_manip.o      : "
	@$(COMPIL) $(FLAGS) -c arrays_manip.F90
	@echo "    ok"

modd_io.o : modd_io.F90 modd_parameters.o
	@echo -n "modd_io.o    : "
	@$(COMPIL) $(FLAGS) -c modd_io.F90
	@echo "    ok"

modd_parameters.o : modd_parameters.F90
	@echo -n "modd_parameters.o    : "
	@$(COMPIL) $(FLAGS) -c modd_parameters.F90
	@echo "    ok"

xrd_getoptions.o : xrd_getoptions.F90 xrd_unix_env.o
	@echo -n "xrd_getoptions.o   : "
	@$(COMPIL) $(FLAGS) -c xrd_getoptions.F90
	@echo "    ok"

xrd_unix_env.o : xrd_unix_env.F90 parkind1.o
	@echo -n "xrd_unix_env.o   : "
	@$(COMPIL) $(FLAGS) -c xrd_unix_env.F90
	@echo "    ok"

parkind1.o : parkind1.F90
	@echo -n "parkind1.o   : "
	@$(COMPIL) $(FLAGS) -c parkind1.F90
	@echo "    ok"





# modlinalg.o : modlinalg.f90 modshare.o
# 	@echo -n "modlinalg.o : "
# 	@$(COMPIL) $(FLAGS) -c modlinalg.f90 -llapack -lblas
# 	@echo "    ok"


#---------------------------------------------------------------------#
#--------------------  Nettoyage des repertoires  --------------------#
clean:
	@clear
	@clear
	@rm -f *~ *.o mod_*.mod
	@rm -f ../$(EXE) ../*~
	@echo "Repertoire propre"
	@echo " "
#---------------------------------------------------------------------#
#--------------------  Nettoyage des repertoires  --------------------#
proper:
	@clear
	@clear
	@rm -f *~ *.o *.mod *.a
	@rm -f ../$(EXE) ../*~ ../res/*
	@echo "Repertoire propre"
	@echo " "
#--------------------------------------------------------------------#
