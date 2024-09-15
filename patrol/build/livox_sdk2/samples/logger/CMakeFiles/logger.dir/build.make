# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/whaltze/rm/patrol/Livox-SDK2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/whaltze/rm/patrol/build/livox_sdk2

# Include any dependencies generated for this target.
include samples/logger/CMakeFiles/logger.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include samples/logger/CMakeFiles/logger.dir/compiler_depend.make

# Include the progress variables for this target.
include samples/logger/CMakeFiles/logger.dir/progress.make

# Include the compile flags for this target's objects.
include samples/logger/CMakeFiles/logger.dir/flags.make

samples/logger/CMakeFiles/logger.dir/main.cpp.o: samples/logger/CMakeFiles/logger.dir/flags.make
samples/logger/CMakeFiles/logger.dir/main.cpp.o: /home/whaltze/rm/patrol/Livox-SDK2/samples/logger/main.cpp
samples/logger/CMakeFiles/logger.dir/main.cpp.o: samples/logger/CMakeFiles/logger.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/whaltze/rm/patrol/build/livox_sdk2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object samples/logger/CMakeFiles/logger.dir/main.cpp.o"
	cd /home/whaltze/rm/patrol/build/livox_sdk2/samples/logger && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT samples/logger/CMakeFiles/logger.dir/main.cpp.o -MF CMakeFiles/logger.dir/main.cpp.o.d -o CMakeFiles/logger.dir/main.cpp.o -c /home/whaltze/rm/patrol/Livox-SDK2/samples/logger/main.cpp

samples/logger/CMakeFiles/logger.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/logger.dir/main.cpp.i"
	cd /home/whaltze/rm/patrol/build/livox_sdk2/samples/logger && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/whaltze/rm/patrol/Livox-SDK2/samples/logger/main.cpp > CMakeFiles/logger.dir/main.cpp.i

samples/logger/CMakeFiles/logger.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/logger.dir/main.cpp.s"
	cd /home/whaltze/rm/patrol/build/livox_sdk2/samples/logger && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/whaltze/rm/patrol/Livox-SDK2/samples/logger/main.cpp -o CMakeFiles/logger.dir/main.cpp.s

# Object files for target logger
logger_OBJECTS = \
"CMakeFiles/logger.dir/main.cpp.o"

# External object files for target logger
logger_EXTERNAL_OBJECTS =

samples/logger/logger: samples/logger/CMakeFiles/logger.dir/main.cpp.o
samples/logger/logger: samples/logger/CMakeFiles/logger.dir/build.make
samples/logger/logger: sdk_core/liblivox_lidar_sdk_static.a
samples/logger/logger: samples/logger/CMakeFiles/logger.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/whaltze/rm/patrol/build/livox_sdk2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable logger"
	cd /home/whaltze/rm/patrol/build/livox_sdk2/samples/logger && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/logger.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
samples/logger/CMakeFiles/logger.dir/build: samples/logger/logger
.PHONY : samples/logger/CMakeFiles/logger.dir/build

samples/logger/CMakeFiles/logger.dir/clean:
	cd /home/whaltze/rm/patrol/build/livox_sdk2/samples/logger && $(CMAKE_COMMAND) -P CMakeFiles/logger.dir/cmake_clean.cmake
.PHONY : samples/logger/CMakeFiles/logger.dir/clean

samples/logger/CMakeFiles/logger.dir/depend:
	cd /home/whaltze/rm/patrol/build/livox_sdk2 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/whaltze/rm/patrol/Livox-SDK2 /home/whaltze/rm/patrol/Livox-SDK2/samples/logger /home/whaltze/rm/patrol/build/livox_sdk2 /home/whaltze/rm/patrol/build/livox_sdk2/samples/logger /home/whaltze/rm/patrol/build/livox_sdk2/samples/logger/CMakeFiles/logger.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : samples/logger/CMakeFiles/logger.dir/depend

